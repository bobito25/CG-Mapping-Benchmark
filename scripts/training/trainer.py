import jax
import jax.numpy as jnp
from chemtrain.trainers import ForceMatching
from chemtrain.learn import max_likelihood
from jax_md import space
from cgbench.core.mapping import _map_single as cg_map_single
from typing import Dict, Any

class CGForceMatching(ForceMatching):
    def __init__(self, init_params, optimizer, energy_fn_template, nbrs_init, model_cg_map, cg_species, atom_masses, freeze_cg=False, empty_bead_penalty_weight=0.0, use_direct_force_mapping=True, learned_species_embedding=False, **kwargs):
        self.model_cg_map = model_cg_map
        self.nbrs_init = nbrs_init
        self.cg_species = jnp.asarray(cg_species, dtype=jnp.int32)
        self.atom_masses = jnp.asarray(atom_masses, dtype=jnp.float32)
        self.freeze_cg = freeze_cg
        self.empty_bead_penalty_weight = empty_bead_penalty_weight
        self.temperature = 1.0
        self.use_direct_force_mapping = use_direct_force_mapping
        self.learned_species_embedding = learned_species_embedding
        self.epoch_cg_gradients = []
        self.saved_cg_gradients = {}
        
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
            
            # Convert R_cg_cart back to fractional coordinates
            inv_box_tensor = jnp.linalg.inv(box_tensor)
            R_cg = jax.vmap(lambda r: jnp.dot(inv_box_tensor, r.T).T)(R_cg_cart)

            # Pass actual CG species and masks for the target CG layer
            cg_mask = jnp.sum(assignments, axis=-2) > 0.0

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

            # Push mapped reference forces into predictions so custom loss can find them
            predictions['Mapped_Target_F'] = F_cg
            predictions['cg_mask'] = cg_mask
            predictions['R_cg'] = R_cg
            predictions['assignments'] = assignments
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
            
            # Extract gradients from outputs (index 3 is the gradient structure)
            grad = outputs[3]
            grad_cg = grad.get('cg_map', None)
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
            return outputs
        self._update_fn = wrapped_update_fn

    def _update(self, batch):
        # Inject current temperature into batch dictionary as a 1D JAX array matching batch_size to prevent multi-GPU sharding error
        batch['temperature'] = jnp.full((batch['R'].shape[0],), self.temperature, dtype=jnp.float32)
        super()._update(batch)
