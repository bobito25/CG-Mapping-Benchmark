import flax.linen as fnn
import jax
import jax.numpy as jnp


@jax.jit
def gumbel_softmax(logits: jnp.ndarray, key: jnp.ndarray, temperature: jnp.ndarray = 1.0) -> jnp.ndarray:
    u = jax.random.uniform(key, shape=logits.shape, minval=1e-7, maxval=1.0 - 1e-7)
    gumbel_noise = -jnp.log(-jnp.log(u))
    y = logits + gumbel_noise
    temp_arr = jnp.maximum(jnp.asarray(temperature), 1e-5)
    while temp_arr.ndim < logits.ndim:
        temp_arr = temp_arr[..., None]
    return jax.nn.softmax(y / temp_arr, axis=-1)


class GumbelCGAssignment(fnn.Module):
    num_cg_beads: int
    initial_mapping: tuple = None
    learned_species_embedding: bool = False
    normalize_atom_embedding: bool = False
    species_embedding_dim: int = 16
    unique_atom_species: tuple = None

    @fnn.compact
    def __call__(self, positions: jnp.ndarray, species: jnp.ndarray, key: jnp.ndarray,
                 atom_masses: jnp.ndarray, deterministic: bool = False, temperature: float = 1.0,
                 return_assignments: bool = False, return_species_features: bool = False) -> jnp.ndarray:
        """
        Assigns each node to a coarse-grained bead using Gumbel-Softmax.

        Args:
            positions: Node positions of shape (num_nodes, 3).
            species: Node species of shape (num_nodes,).
            key: JAX random key for sampling Gumbel noise.
            atom_masses: Array mapping species index to mass.
            deterministic: If True, bypass Gumbel-Softmax noise and assign nodes directly.
            temperature: Temperature for Gumbel-Softmax.
            return_assignments: If True, return (c_map, assignments).
            return_species_features: If True, compute and return learned combined species features.

        Returns:
            c_map: Coarse-graining mapping matrix of shape (num_cg_beads, num_nodes).
        """
        num_atoms = species.shape[-1]
        one_hot_indices = jax.nn.one_hot(jnp.arange(num_atoms), num_atoms)
        node_attrs = jnp.broadcast_to(one_hot_indices, species.shape + (num_atoms,))

        if self.initial_mapping is not None:
            initial_map_arr = jnp.array(self.initial_mapping, dtype=jnp.int32)
            
            def custom_init(key, shape, dtype=jnp.float32):
                one_hot_map = jax.nn.one_hot(initial_map_arr, self.num_cg_beads, dtype=dtype)
                return one_hot_map * 5.0

            logits = fnn.Dense(self.num_cg_beads, kernel_init=custom_init, bias_init=fnn.initializers.zeros)(node_attrs)
        else:
            logits = fnn.Dense(self.num_cg_beads)(node_attrs)

        if deterministic:
            hard_indices = jnp.argmax(logits, axis=-1)
            assignments = jax.nn.one_hot(hard_indices, self.num_cg_beads, dtype=logits.dtype)
        else:
            soft_assignments = gumbel_softmax(logits, key, temperature)  # [ num_nodes, num_cg_beads ]
            hard_indices = jnp.argmax(soft_assignments, axis=-1)
            hard_assignments = jax.nn.one_hot(hard_indices, self.num_cg_beads, dtype=soft_assignments.dtype)  # [ num_nodes, num_cg_beads ]
            # use straight-through estimator to allow gradient flow through soft assignments while using hard assignments in the forward pass
            assignments = jax.lax.stop_gradient(hard_assignments - soft_assignments) + soft_assignments

        # Lookup masses based on node index
        node_masses = jnp.broadcast_to(atom_masses, species.shape) # Shape: [..., num_nodes]

        # Weight assignments by mass
        weighted_assignments = assignments * node_masses[..., None]
        bead_masses = jnp.maximum(jnp.sum(weighted_assignments, axis=-2, keepdims=True), 1e-3)
        
        # Transpose the last two dimensions to get [..., num_cg_beads, num_nodes]
        c_map = jnp.swapaxes(weighted_assignments, -1, -2) / jnp.swapaxes(bead_masses, -1, -2)

        species_features = None
        if self.learned_species_embedding or return_species_features:
            if self.unique_atom_species is not None:
                unique_species = jnp.array(self.unique_atom_species, dtype=species.dtype)
            else:
                flat_species = species.reshape(-1)
                unique_species = jnp.unique(flat_species)

            num_atom_types = unique_species.shape[0]
            atom_type_indices = jnp.argmax(species[..., :, None] == unique_species[None, :], axis=-1)
            one_hot_atom_types = jax.nn.one_hot(atom_type_indices, num_atom_types, dtype=assignments.dtype)

            assignments_t = jnp.swapaxes(assignments, -1, -2) # [..., num_cg_beads, num_nodes]
            bead_atom_types = jnp.matmul(assignments_t, one_hot_atom_types) # [..., num_cg_beads, num_atom_types]

            atom_type_embeddings = self.param(
                'atom_type_embeddings',
                fnn.initializers.normal(stddev=0.1),
                (num_atom_types, self.species_embedding_dim)
            )
            if self.normalize_atom_embedding:
                norm = jnp.linalg.norm(atom_type_embeddings, axis=-1, keepdims=True)
                atom_type_embeddings = atom_type_embeddings / (norm + 1e-8)

            species_features = jnp.matmul(bead_atom_types, atom_type_embeddings) # [..., num_cg_beads, species_embedding_dim]

        if return_assignments and return_species_features:
            return c_map, assignments, species_features
        elif return_assignments:
            return c_map, assignments
        elif return_species_features:
            return c_map, species_features
        return c_map