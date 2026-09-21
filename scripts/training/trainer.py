import jax
import jax.numpy as jnp
import numpy as onp
from chemtrain.trainers import ForceMatching
from chemtrain.learn import max_likelihood
from jax_md import space
from cgbench.core.mapping import _map_single as cg_map_single
from typing import Dict, Any
from nan_detector import check_array_jit, check_pairwise_distances, inspect_gradients, inspect_losses

class CGForceMatching(ForceMatching):
    def __init__(self, init_params, optimizer, energy_fn_template, nbrs_init, model_cg_map, cg_species, atom_masses, freeze_cg=False, empty_bead_penalty_weight=0.0, intra_bead_distance_weight=0.0, use_direct_force_mapping=True, learned_species_embedding=False, debug_nans=False, **kwargs):
        self.model_cg_map = model_cg_map
        self.nbrs_init = nbrs_init
        self.cg_species = jnp.asarray(cg_species, dtype=jnp.int32)
        self.atom_masses = jnp.asarray(atom_masses, dtype=jnp.float32)
        self.freeze_cg = freeze_cg
        self.empty_bead_penalty_weight = empty_bead_penalty_weight
        self.intra_bead_distance_weight = intra_bead_distance_weight
        self.temperature = 1.0
        self.use_direct_force_mapping = use_direct_force_mapping
        self.learned_species_embedding = learned_species_embedding
        self.debug_nans = debug_nans
        self.epoch_cg_gradients = []
        self.saved_cg_gradients = {}
        self._step_counter = 0
        
        # Wrap update_fn of nbrs_init to stop gradients on position
        original_update_fn = nbrs_init.update_fn
        nbrs_init_stopped = nbrs_init.set(
            update_fn=lambda pos, *args, **kwargs: original_update_fn(jax.lax.stop_gradient(pos), *args, **kwargs)
        )
        
        # Initialize the standard ForceMatching first to get the default model and loss
        super().__init__(init_params, optimizer, energy_fn_template, nbrs_init_stopped, **kwargs)

        # Extract chemtrain's default functions
        base_model = self.batched_model
        base_loss_fn = self._loss_fn

        # Create a wrapper model that mutates the batch inside the JAX trace
        def cg_batched_model(params: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
            # batch contains atomistic data:
            # - 'R': Atomistic coordinates (batch_size, n_atoms, 3)
            # - 'F': Target atomistic forces (batch_size, n_atoms, 3)
            # - 'box': Simulation box dimensions (batch_size, 3)
            # - 'species': Atom species (batch_size, n_atoms)
            # - 'mask': Atom mask padding (batch_size, n_atoms)

            # Use batch-specific PRNGKey if available, otherwise fallback
            if 'rng' in batch:
                prng_key = batch['rng'][0]
            else:
                prng_key = params.get('Dropout_RNG_key', jax.random.PRNGKey(0))

            # Deterministic if freeze_cg is True, or if batch has no rng (validation/evaluation/predictions)
            deterministic = self.freeze_cg or ('rng' not in batch)

            # Retrieve temperature from batch or default to 1.0 (not used during deterministic evaluation)
            temperature = batch.get('temperature', jnp.array(1.0, dtype=jnp.float32))

            # Compute CG assignment
            if self.learned_species_embedding:
                c_map, assignments, species_features = self.model_cg_map.apply(
                    {'params': params['cg_map']}, 
                    batch['R'], batch['species'], prng_key,
                    self.atom_masses,
                    deterministic=deterministic,
                    temperature=temperature,
                    return_assignments=True,
                    return_species_features=True
                )
                cg_species = species_features
            else:
                c_map, assignments = self.model_cg_map.apply(
                    {'params': params['cg_map']}, 
                    batch['R'], batch['species'], prng_key,
                    self.atom_masses,
                    deterministic=deterministic,
                    temperature=temperature,
                    return_assignments=True
                )
                cg_species = jnp.tile(self.cg_species, (batch['R'].shape[0], 1))
                species_features = None

            if self.debug_nans:
                c_map = check_array_jit("c_map", c_map)
                assignments = check_array_jit("assignments", assignments)
                if species_features is not None:
                    species_features = check_array_jit("species_features", species_features)

            # Convert R to Cartesian space
            box_tensor = batch["box"][0]
            if box_tensor.ndim != 2:
                box_tensor = jnp.eye(box_tensor.shape[0]) * box_tensor
            
            # batch['R'] is shape (batch_size, n_atoms, 3)
            # R_cart = R_frac @ box_tensor^T
            R_cart = jax.vmap(lambda r: jnp.dot(box_tensor, r.T).T)(batch['R'])
            
            displacement_fn_cart, shift_fn_cart = space.periodic_general(
                box=box_tensor, fractional_coordinates=False
            )

            if self.use_direct_force_mapping:
                # Map AT -> CG coordinates dynamically in Cartesian space
                R_cg_cart, _ = jax.vmap(cg_map_single, in_axes=(0, None, None, 0, 0))(
                    (R_cart, jnp.zeros_like(R_cart)), shift_fn_cart, displacement_fn_cart, c_map, c_map
                )
                # Map forces using assignments directly
                assignments_t = jnp.swapaxes(assignments, -1, -2)
                F_cg = jnp.einsum("BIn, Bnd -> BId", assignments_t, batch['F'])
            else:
                # Map AT -> CG for both R and F dynamically in Cartesian space
                R_cg_cart, F_cg = jax.vmap(cg_map_single, in_axes=(0, None, None, 0, 0))(
                    (R_cart, batch['F']), shift_fn_cart, displacement_fn_cart, c_map, c_map
                )
            
            # Pass actual CG species and masks for the target CG layer
            cg_mask = jnp.sum(assignments, axis=-2) > 0.0

            if self.debug_nans:
                R_cg_cart = check_array_jit("R_cg_cart", R_cg_cart)
                F_cg = check_array_jit("F_cg", F_cg)
                check_pairwise_distances("R_cg_cart", R_cg_cart, mask=cg_mask)

            # Convert R_cg_cart back to fractional coordinates
            inv_box_tensor = jnp.linalg.inv(box_tensor)
            R_cg = jax.vmap(lambda r: jnp.dot(inv_box_tensor, r.T).T)(R_cg_cart)

            # Exclude temperature from cg_batch to prevent vmap ranking errors in base_model
            clean_batch = {k: v for k, v in batch.items() if k != 'temperature'}

            cg_batch = {
                **clean_batch, 
                'R': R_cg,
                'F': F_cg, 
                'species': cg_species,
                'mask': cg_mask,
            }

            # Run chemtrain's standard model on the CG proxy batch
            predictions = base_model(params['mace'], cg_batch)

            if self.debug_nans:
                if 'F' in predictions:
                    predictions['F'] = check_array_jit("mace_predictions['F']", predictions['F'])
                    is_nan_F = jnp.isnan(predictions['F'])
                    def _print_nan_bead_info():
                        nan_bead_mask = jnp.any(is_nan_F, axis=-1) # (32, 10)
                        active_nans = jnp.sum(nan_bead_mask & cg_mask)
                        empty_nans = jnp.sum(nan_bead_mask & (~cg_mask))
                        
                        # Find first frame with NaN
                        frame_has_nan = jnp.any(nan_bead_mask, axis=-1)
                        b_idx = jnp.argmax(frame_has_nan)
                        
                        frame_nan_beads = nan_bead_mask[b_idx]
                        frame_cg_mask = cg_mask[b_idx]
                        frame_counts = jnp.sum(assignments[b_idx], axis=-2)
                        frame_pos = R_cg_cart[b_idx]
                        
                        nan_bead_indices = jnp.where(frame_nan_beads, jnp.arange(10), -1)

                        jax.debug.print(
                            "\n" + "="*80 + "\n"
                            "[NAN FORCE PINPOINT]\n"
                            "Total NaN beads in batch: {tot} (Active: {act}, Empty: {emp})\n"
                            "First Corrupted Batch Index: {b}\n"
                            "Corrupted Bead Indices: {bead_idxs}\n"
                            "Mask for this frame (cg_mask): {mask}\n"
                            "Atom counts per bead (sum of assignments): {counts}\n"
                            "Positions of beads:\n{pos}\n"
                            + "="*80,
                            tot=jnp.sum(nan_bead_mask),
                            act=active_nans,
                            emp=empty_nans,
                            b=b_idx,
                            bead_idxs=nan_bead_indices,
                            mask=frame_cg_mask,
                            counts=frame_counts,
                            pos=frame_pos,
                        )
                    jax.lax.cond(jnp.any(is_nan_F), _print_nan_bead_info, lambda: None)
                if 'energy' in predictions:
                    predictions['energy'] = check_array_jit("mace_predictions['energy']", predictions['energy'])

            # Cleanly mask predicted forces so unassigned/empty beads cannot inject NaNs into the loss
            if 'F' in predictions:
                predictions['F'] = jnp.where(cg_mask[:, :, None], predictions['F'], 0.0)

            # Push mapped reference forces into predictions so custom loss can find them
            predictions['Mapped_Target_F'] = F_cg
            predictions['cg_mask'] = cg_mask
            predictions['R_cg'] = R_cg
            predictions['assignments'] = assignments

            # Compute intra-bead spatial dispersion if weight is positive
            if self.intra_bead_distance_weight > 0.0:
                def frame_dispersion(r_cg, r_at, assign):
                    disp = jax.vmap(lambda r_bead: jax.vmap(lambda r_node: displacement_fn_cart(r_node, r_bead))(r_at))(r_cg)
                    dist_sq = jnp.sum(disp ** 2, axis=-1)  # (num_cg_beads, n_atoms)
                    assign_t = jnp.swapaxes(assign, -1, -2)  # (num_cg_beads, n_atoms)
                    bead_counts = jnp.sum(assign_t, axis=-1)  # (num_cg_beads,)
                    bead_sq_sum = jnp.sum(assign_t * dist_sq, axis=-1)  # (num_cg_beads,)
                    safe_counts = jnp.maximum(bead_counts, 1e-4)
                    bead_var = bead_sq_sum / safe_counts  # (num_cg_beads,)
                    active_mask = bead_counts > 0.5
                    num_active = jnp.maximum(jnp.sum(active_mask), 1.0)
                    return jnp.sum(bead_var * active_mask) / num_active

                frame_disp_fn = lambda r_cg, r_at, assign: frame_dispersion(r_cg, r_at, assign)
                batch_dispersions = jax.vmap(frame_disp_fn)(R_cg_cart, R_cart, assignments)
                predictions['intra_bead_dispersion'] = batch_dispersions

            return predictions

        # Create a custom loss that looks at the mapped targets instead of the original batch targets
        def cg_loss_fn(predictions, original_batch):
            # We construct a proxy batch that contains the dynamically mapped F_cg
            proxy_batch = {**original_batch, 'F': predictions['Mapped_Target_F']}
            loss_val, errors = base_loss_fn(predictions, proxy_batch)
            
            # Retrieve mask of shape (batch_size, num_cg_beads)
            cg_mask = predictions['cg_mask']
            total_beads = cg_mask.size
            num_active_beads = jnp.sum(cg_mask)
            
            # Scale factor to divide by active beads instead of total beads
            scale_factor = total_beads / (num_active_beads + 1e-8)
            
            scaled_F_error = errors['F'] * scale_factor
            corrected_loss = loss_val - errors['F'] + scaled_F_error
            
            errors['F'] = scaled_F_error

            # Add empty bead penalty if weight is positive and assignments are present
            if 'assignments' in predictions and self.empty_bead_penalty_weight > 0.0:
                assignments = predictions['assignments']
                # assignments shape: (batch_size, num_nodes, num_cg_beads)
                # Sum assignments over nodes to get total assignment count per bead: (batch_size, num_cg_beads)
                bead_counts = jnp.sum(assignments, axis=-2)
                # Penalty is relu(1.0 - bead_counts)
                penalty_per_bead = jax.nn.relu(1.0 - bead_counts)
                # Mean over batch of the sum of penalties over beads
                empty_bead_penalty = jnp.mean(jnp.sum(penalty_per_bead, axis=-1))
                
                loss_penalty = self.empty_bead_penalty_weight * empty_bead_penalty
                corrected_loss = corrected_loss + loss_penalty
                errors['empty_bead_penalty'] = empty_bead_penalty
                errors['empty_bead_loss'] = loss_penalty

            # Add intra-bead distance penalty if weight is positive and dispersion is computed
            if 'intra_bead_dispersion' in predictions and self.intra_bead_distance_weight > 0.0:
                intra_disp = jnp.mean(predictions['intra_bead_dispersion'])
                intra_loss = self.intra_bead_distance_weight * intra_disp
                corrected_loss = corrected_loss + intra_loss
                errors['intra_bead_dispersion'] = intra_disp
                errors['intra_bead_loss'] = intra_loss

            if self.debug_nans:
                scaled_F_error = check_array_jit("scaled_F_error", scaled_F_error)
                corrected_loss = check_array_jit("corrected_loss", corrected_loss)

            return corrected_loss, errors

        # Overwrite the tracked references
        self.batched_model = cg_batched_model
        self._loss_fn = cg_loss_fn

        # Recompile the _update_fn and _evaluate_fn to use new cg_batched_model instead
        if self._disable_shmap:
            self._update_fn = max_likelihood.pmap_update_fn(
                self.batched_model, self._loss_fn, optimizer, penalty_fn=kwargs.get('penalty_fn'))
            self._evaluate_fn = None
        else:
            self._update_fn = max_likelihood.shmap_update_fn(
                self.batched_model, self._loss_fn, optimizer, penalty_fn=kwargs.get('penalty_fn'))
            self._evaluate_fn = max_likelihood.shmap_loss_fn(
                self.batched_model, self._loss_fn, penalty_fn=kwargs.get('penalty_fn'))

        original_train_update_fn = self._update_fn
        def wrapped_update_fn(params, opt_state, batch, per_target=True):
            outputs = original_train_update_fn(params, opt_state, batch, per_target=per_target)
            
            # Extract outputs: (new_params, new_opt_state, loss, grad, per_target_loss)
            loss = outputs[2]
            grad = outputs[3]
            per_target_loss = outputs[4] if len(outputs) > 4 else None

            # Extract gradients for tracking
            grad_cg = grad.get('cg_map', None) if isinstance(grad, dict) else None
            dense_keys = []
            if grad_cg is not None:
                dense_keys = [k for k in grad_cg.keys() if 'Dense' in k]
                if dense_keys:
                    dense_grad = grad_cg[dense_keys[0]]
                    kernel_grad = jax.device_get(dense_grad.get('kernel', 0.0))
                    bias_grad = jax.device_get(dense_grad.get('bias', 0.0))
                    self.epoch_cg_gradients.append({
                        'kernel': kernel_grad,
                        'bias': bias_grad
                    })

            # LIGHTWEIGHT CHECK (Always run):
            # Check if loss is finite
            loss_val = float(onp.asarray(jax.device_get(loss)))
            is_loss_corrupted = not onp.isfinite(loss_val)

            # Check if mapping gradients contain NaNs (only inspects ~200 floats in Dense layer)
            cg_grad_corrupted = False
            if grad_cg is not None and dense_keys:
                dense_grad = grad_cg[dense_keys[0]]
                k_grad = dense_grad.get('kernel', None)
                if k_grad is not None:
                    k_arr = onp.asarray(jax.device_get(k_grad))
                    if not onp.all(onp.isfinite(k_arr)):
                        cg_grad_corrupted = True

            # Trigger detailed diagnostics if loss or cg_grad is corrupted, or if debug_nans is enabled
            if is_loss_corrupted or cg_grad_corrupted or self.debug_nans:
                inspect_losses(loss, per_target_loss, step_idx=self._step_counter)
                has_nan_or_inf, _ = inspect_gradients(grad, step_idx=self._step_counter)
                if is_loss_corrupted or cg_grad_corrupted or has_nan_or_inf:
                    raise FloatingPointError(
                        f"[FATAL] NaN or infinite value detected at step {self._step_counter}! Halting training immediately."
                    )

            self._step_counter += 1
            return outputs
        self._update_fn = wrapped_update_fn

    def evaluate_per_bead_forces(self, dataset, params=None, batch_size=32):
        return evaluate_per_bead_forces(self, dataset, params=params, batch_size=batch_size)

    def _update(self, batch):
        # Inject current temperature into batch dictionary as a 1D JAX array matching batch_size to prevent multi-GPU sharding error
        batch['temperature'] = jnp.full((batch['R'].shape[0],), self.temperature, dtype=jnp.float32)
        super()._update(batch)


def evaluate_per_bead_forces(trainer, dataset, params=None, batch_size=32):
    """
    Evaluate per-bead force MSE loss mean and variance across the provided dataset.

    Parameters
    ----------
    trainer : ForceMatching or CGForceMatching
        The trainer instance.
    dataset : dict
        Dataset containing 'R', 'F', etc.
    params : dict, optional
        Model parameters. If None, uses trainer.params.
    batch_size : int, optional
        Batch size for prediction.

    Returns
    -------
    mean_per_bead : np.ndarray of shape (num_beads,)
        Mean force MSE loss for each bead across all frames.
    var_per_bead : np.ndarray of shape (num_beads,)
        Variance of force MSE loss for each bead across all frames.
    """
    import numpy as onp
    from jax import tree_util

    if params is None:
        params = getattr(trainer, "params", None)
        if params is None and hasattr(trainer, "best_params"):
            params = trainer.best_params

    preds = trainer.predict(dataset, params, batch_size=batch_size)
    preds = tree_util.tree_map(onp.asarray, preds)

    pred_F = preds["F"]
    if "Mapped_Target_F" in preds:
        ref_F = preds["Mapped_Target_F"]
    elif "F" in dataset:
        ref_F = onp.asarray(dataset["F"])
    else:
        raise ValueError("Reference forces not found in predictions or dataset.")

    n_samples = min(len(pred_F), len(ref_F))
    pred_F = pred_F[:n_samples]
    ref_F = ref_F[:n_samples]

    # Force squared error per bead per sample: sum over spatial 3D dimensions
    diff = pred_F - ref_F
    sq_err = onp.sum(diff ** 2, axis=-1)  # shape: (n_samples, n_beads)

    if "mask" in preds:
        mask = onp.asarray(preds["mask"])[:n_samples]
        n_beads = sq_err.shape[1]
        mean_per_bead = onp.zeros(n_beads, dtype=onp.float32)
        var_per_bead = onp.zeros(n_beads, dtype=onp.float32)
        for b in range(n_beads):
            valid_samples = sq_err[:, b][mask[:, b] > 0]
            if len(valid_samples) > 0:
                mean_per_bead[b] = onp.mean(valid_samples)
                var_per_bead[b] = onp.var(valid_samples)
    else:
        mean_per_bead = onp.mean(sq_err, axis=0)
        var_per_bead = onp.var(sq_err, axis=0)

    return mean_per_bead, var_per_bead

