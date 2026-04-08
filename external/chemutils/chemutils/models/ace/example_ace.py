#!/usr/bin/env python3
"""
Example usage of ACE model.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as onp
from jax_md import space, partition

from model import ace_neighborlist_pp


def create_simple_system(n_atoms=20, box_size=8.0):
    """Create a simple test system."""
    # Random positions in a box
    positions = jnp.array(onp.random.rand(n_atoms, 3) * box_size)
    
    # All atoms are the same species (index 0)
    species = jnp.zeros(n_atoms, dtype=jnp.int32)
    
    return positions, species


def main():
    """Main example function."""
    print("ACE Model Example")
    print("=" * 50)
    
    # Create test system
    positions, species = create_simple_system(n_atoms=20)
    print(f"Created system with {len(positions)} atoms")
    
    # Create displacement function (free space)
    displacement_fn = space.free()
    
    # Create neighbor list
    r_cutoff = 3.0
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size=10.0,
        r_cutoff=r_cutoff,
        dr_threshold=0.5,
        format=partition.Dense
    )
    
    # Initialize neighbor list
    nbrs = neighbor_fn.allocate(positions)
    print(f"Neighbor list created with cutoff {r_cutoff}")
    
    # Create ACE model
    print("\nCreating ACE model...")
    ace_init, ace_apply = ace_neighborlist_pp(
        displacement=displacement_fn,
        r_cutoff=r_cutoff,
        n_species=1,  # Only one species
        positions_test=positions,
        neighbor_test=nbrs,
        # ACE-specific parameters
        n_max=6,           # Maximum radial basis degree
        l_max=3,           # Maximum angular momentum
        correlation_order=3,  # Many-body correlation order
        include_pair=True,    # Include pair potential
        include_onebody=True, # Include one-body reference energy
    )
    
    # Initialize model parameters
    print("Initializing model parameters...")
    key = jax.random.PRNGKey(42)
    params = ace_init(key, positions, nbrs, species)
    
    # Count parameters
    total_params = sum(p.size for p in jax.tree_leaves(params))
    print(f"Total number of parameters: {total_params:,}")
    
    # Compute energy
    print("\nComputing energy...")
    energy = ace_apply(params, key, positions, nbrs, species)
    print(f"Total energy: {energy:.6f} eV")
    
    # Test energy conservation under translation
    print("\nTesting energy conservation...")
    translation = jnp.array([1.0, 0.0, 0.0])
    positions_translated = positions + translation
    nbrs_translated = neighbor_fn.update(positions_translated, nbrs)
    energy_translated = ace_apply(params, key, positions_translated, nbrs_translated, species)
    
    energy_diff = abs(energy - energy_translated)
    print(f"Energy difference after translation: {energy_diff:.2e} eV")
    
    if energy_diff < 1e-6:
        print("✓ Energy is translationally invariant!")
    else:
        print("⚠ Energy is not translationally invariant (this might be expected for some models)")
    
    # Test with different configurations
    print("\nTesting with different configurations...")
    for i in range(3):
        # Random displacement
        displacement = jnp.array(onp.random.randn(3)) * 0.1
        positions_new = positions + displacement
        nbrs_new = neighbor_fn.update(positions_new, nbrs)
        energy_new = ace_apply(params, key, positions_new, nbrs_new, species)
        print(f"Configuration {i+1}: Energy = {energy_new:.6f} eV")
    
    print("\n🎉 Example completed successfully!")
    print("\nKey features of this ACE implementation:")
    print("- Linear model with many-body correlations")
    print("- Radial basis functions with envelope cutoff")
    print("- Spherical harmonics for angular dependence")
    print("- Tensor products for many-body interactions")
    print("- Species-specific parameters")
    print("- Compatible with JAX-MD ecosystem")


if __name__ == "__main__":
    main()
