#!/usr/bin/env python3
"""
Test script for ACE implementation.
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn
import numpy as onp

from model import ACE, ace_neighborlist_pp


from jax_md import space, partition


def test_ace_model():
    """Test basic ACE model functionality."""
    print("Testing ACE model...")
    
    # Create a simple test system
    n_atoms = 10
    positions = jnp.array(onp.random.randn(n_atoms, 3) * 2.0)
    species = jnp.zeros(n_atoms, dtype=jnp.int32)  # All same species
    
    # Create displacement function
    displacement_fn = space.free()
    
    # Create neighbor list
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size=10.0,
        r_cutoff=3.0,
        dr_threshold=0.5,
        format=partition.Dense
    )
    
    # Initialize neighbor list
    nbrs = neighbor_fn.allocate(positions)
    
    # Create ACE model
    ace_init, ace_apply = ace_neighborlist_pp(
        displacement=displacement_fn,
        r_cutoff=3.0,
        n_species=1,
        positions_test=positions,
        neighbor_test=nbrs,
        n_max=4,
        l_max=2,
        correlation_order=2,
        include_pair=True,
        include_onebody=True
    )
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = ace_init(key, positions, nbrs, species)
    
    print(f"Model initialized successfully!")
    print(f"Number of parameters: {sum(p.size for p in jax.tree_leaves(params))}")
    
    # Test forward pass
    energy = ace_apply(params, key, positions, nbrs, species)
    print(f"Energy computed: {energy}")
    
    # Test with different positions
    positions2 = positions + jnp.array([0.1, 0.0, 0.0])
    nbrs2 = neighbor_fn.update(positions2, nbrs)
    energy2 = ace_apply(params, key, positions2, nbrs2, species)
    print(f"Energy after displacement: {energy2}")
    
    print("✓ ACE model test passed!")



def test_radial_basis():
    """Test radial basis layer."""
    from ace_layers import RadialBasisLayer
    
    def radial_basis_fn(r):
        radial_basis = RadialBasisLayer(n_max=4, l_max=2, rcut=3.0)
        return radial_basis(r)
    
    # Transform the function
    radial_basis_init, radial_basis_apply = hk.transform(radial_basis_fn)
    
    # Test data
    r = jnp.linspace(0.5, 2.5, 10)
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = radial_basis_init(key, r)
    
    # Apply function
    Rnl = radial_basis_apply(params, key, r)
    print(f"Radial basis shape: {Rnl.shape}")
    
    return Rnl


def test_tensor_product():
    """Test tensor product layer."""
    from ace_layers import ACETensorProductLayer
    
    def tensor_product_fn(Rnl, Ylm, node_species, senders, receivers):
        tensor_layer = ACETensorProductLayer(
            correlation_order=2,
            n_max=4,
            l_max=2,
            num_species=1
        )
        return tensor_layer(Rnl, Ylm, node_species, senders, receivers)
    
    # Transform the function
    tensor_init, tensor_apply = hk.transform(tensor_product_fn)
    
    # Create dummy data
    n_edges = 20
    n_nodes = 10
    Rnl = jnp.random.randn(n_edges, 4 * 3)  # n_max * (l_max + 1)
    Ylm = jnp.random.randn(n_edges, 9)  # (l_max + 1)^2
    node_species = jnp.zeros(n_nodes, dtype=jnp.int32)
    senders = jnp.random.randint(0, n_nodes, (n_edges,))
    receivers = jnp.random.randint(0, n_nodes, (n_edges,))
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = tensor_init(key, Rnl, Ylm, node_species, senders, receivers)
    
    # Apply function
    B = tensor_apply(params, key, Rnl, Ylm, node_species, senders, receivers)
    print(f"Tensor product output shape: {B.shape}")
    
    return B, node_species


def test_readout(B, node_species):
    """Test readout layer."""
    from ace_layers import ACEReadoutLayer
    
    def readout_fn(B, node_species):
        readout = ACEReadoutLayer(num_species=1, num_basis=B.shape[1])
        return readout(B, node_species)
    
    # Transform the function
    readout_init, readout_apply = hk.transform(readout_fn)
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    params = readout_init(key, B, node_species)
    
    # Apply function
    energies = readout_apply(params, key, B, node_species)
    print(f"Readout output shape: {energies.shape}")
    
    return energies


def test_ace_components():
    """Test individual ACE components."""
    print("\nTesting ACE components...")
    
    # Test radial basis
    Rnl = test_radial_basis()
    
    # Test tensor product layer
    B, node_species = test_tensor_product()
    
    # Test readout layer
    energies = test_readout(B, node_species)
    
    print("✓ ACE components test passed!")


if __name__ == "__main__":
    test_ace_components()
    test_ace_model()
    print("\n🎉 All tests passed!")
