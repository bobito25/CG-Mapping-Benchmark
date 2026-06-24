import jax
import jax.numpy as jnp
from chemtrain.trainers import ForceMatching
from chemtrain.learn import max_likelihood
from jax_md import space
from cgbench.core.mapping import _map_single as cg_map_single
from typing import Dict, Any

class CGForceMatching(ForceMatching):
    def __init__(self, init_params, optimizer, energy_fn_template, nbrs_init, model_cg_map, cg_species, atom_masses, freeze_cg=False, **kwargs):
        self.model_cg_map = model_cg_map
        self.nbrs_init = nbrs_init
        self.cg_species = jnp.asarray(cg_species, dtype=jnp.int32)
        self.atom_masses = jnp.asarray(atom_masses, dtype=jnp.float32)
        self.freeze_cg = freeze_cg
        self.temperature = 1.0
        
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
            c_map = self.model_cg_map.apply(
                {'params': params['cg_map']}, 
                batch['R'], batch['species'], prng_key,
                self.atom_masses,
                deterministic=deterministic,
                temperature=temperature
            )

            displacement_fn_X, shift_fn_X = space.periodic_general(
                box=batch["box"][0], fractional_coordinates=True
            )

            # Map AT -> CG for both R and F dynamically
            R_cg, F_cg = jax.vmap(cg_map_single, in_axes=(0, None, None, 0, 0))(
                (batch['R'], batch['F']), shift_fn_X, displacement_fn_X, c_map, c_map
            )

            # Pass actual CG species and masks for the target CG layer
            cg_species = jnp.tile(self.cg_species, (R_cg.shape[0], 1))
            cg_mask = jnp.sum(c_map, axis=-1) > 0.0

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

    def _update(self, batch):
        # Inject current temperature into batch dictionary as a JAX array to prevent re-compilation
        batch['temperature'] = jnp.array(self.temperature, dtype=jnp.float32)
        super()._update(batch)
