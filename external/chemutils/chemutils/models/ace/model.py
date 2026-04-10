import functools
from copy import deepcopy

import math
from typing import Callable, Optional, Union, Set
from collections import OrderedDict

import numpy as onp

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

import jax.nn as jax_nn

from typing import Tuple, Any, Callable


from utils import safe_norm
from layers import (
    EquivariantProductBasisLayer,
    InteractionLayer,
    LinearNodeEmbeddingLayer,
    LinearReadoutLayer,
    NonLinearReadoutLayer,
)
from ace_layers import (
    ACETensorProductLayer,
    ACEReadoutLayer,
    OneBodyLayer,
    PairPotentialLayer,
)
from jax_md_mod.model.layers import (
    RadialBesselLayer,
)

from chemutils.models.layers import AtomicEnergyLayer, CELLI

from jax_md import space, partition, nn, util as md_util
from jax_md_mod.model import sparse_graph


class ACE(hk.Module):
    """Atomic Cluster Expansion (ACE) model.
    
    ACE is a linear model that uses many-body correlations through tensor products
    of radial basis functions and spherical harmonics.
    """
    
    def __init__(
        self,
        *,
        num_species: int,
        n_max: int = 8,  # Maximum radial basis degree
        l_max: int = 3,  # Maximum angular momentum
        correlation_order: int = 3,  # Many-body correlation order
        rcut: float = 5.0,  # Cutoff radius
        r0: float = 0.0,  # Inner cutoff
        polynomial_type: str = "chebyshev",  # Type of radial polynomials
        envelope_type: str = "poly",  # Type of envelope function
        envelope_p: int = 6,  # Envelope polynomial order
        include_pair: bool = True,  # Include pair potential
        include_onebody: bool = True,  # Include one-body reference energy
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        
        self.num_species = num_species
        self.n_max = n_max
        self.l_max = l_max
        self.correlation_order = correlation_order
        self.rcut = rcut
        self.r0 = r0
        self.include_pair = include_pair
        self.include_onebody = include_onebody
        
        # Radial basis functions
        self.radial_basis = RadialBesselLayer(
            cutoff=rcut,
            num_radial=n_max,
            envelope_p=envelope_p
        )
        
        # Spherical harmonics (using e3nn)
        self.spherical_harmonics = e3nn.spherical_harmonics(range(l_max + 1))
        
        # Tensor product layer for many-body correlations
        self.tensor_product = ACETensorProductLayer(
            correlation_order=correlation_order,
            n_max=n_max,
            l_max=l_max,
            num_species=num_species,
            name="tensor_product"
        )
        
        # Readout layer
        self.readout = ACEReadoutLayer(
            num_species=num_species,
            num_basis=self._get_num_basis(),
            name="readout"
        )
        
        # Optional components
        if include_onebody:
            self.onebody = OneBodyLayer(num_species, name="onebody")
        
        if include_pair:
            self.pair_potential = PairPotentialLayer(
                num_species=num_species,
                rcut=rcut,
                n_radial=n_max,
                name="pair_potential"
            )
    
    def _get_num_basis(self) -> int:
        """Calculate total number of basis functions."""
        # Simplified calculation - in practice, this would be more sophisticated
        num_basis = 0
        
        # Single-body terms
        num_basis += self.n_max * (self.l_max + 1)
        
        # Two-body terms (simplified)
        num_basis += self.n_max * (self.l_max + 1) * (self.n_max - 1) * (self.l_max + 1)
        
        return num_basis
    
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes]
        node_mask: Optional[jnp.ndarray] = None,  # [n_nodes]
        is_training: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of ACE model.
        
        Args:
            vectors: Edge vectors
            senders: Sender node indices
            receivers: Receiver node indices
            node_species: Species of each node
            node_mask: Mask for valid nodes
            is_training: Whether in training mode
            
        Returns:
            energies: Per-atom energies [n_nodes]
            features: Many-body basis functions [n_nodes, num_basis]
        """
        if node_mask is None:
            node_mask = jnp.ones(node_species.shape[0], dtype=jnp.bool_)
        
        # Compute distances
        r = safe_norm(vectors.array, axis=-1)  # [n_edges]
        
        # Evaluate radial basis functions
        Rnl = self.radial_basis(r)  # [n_edges, n_max]
        
        # Evaluate spherical harmonics
        Ylm = self.spherical_harmonics(vectors)  # [n_edges, num_harmonics]
        
        # Compute many-body tensor products
        B = self.tensor_product(
            Rnl=Rnl,
            Ylm=Ylm.array,  # Convert to regular array
            node_species=node_species,
            senders=senders,
            receivers=receivers
        )  # [n_nodes, num_basis]
        
        # Compute per-atom energies
        E_ace = self.readout(B, node_species)  # [n_nodes]
        
        # Add one-body reference energy
        if self.include_onebody:
            E_onebody = self.onebody(node_species)
            E_ace = E_ace + E_onebody
        
        # Add pair potential
        if self.include_pair:
            E_pair = self.pair_potential(
                r=r,
                senders=senders,
                receivers=receivers,
                node_species=node_species
            )
            E_ace = E_ace + E_pair
        
        # Apply mask
        E_ace = E_ace * node_mask
        
        return E_ace, B


def ace_neighborlist_pp(displacement: space.DisplacementFn,
                        r_cutoff: float,
                        n_species: int = 100,
                        positions_test: jnp.ndarray = None,
                        neighbor_test: partition.NeighborList = None,
                        max_edge_multiplier: float = 1.25,
                        max_edges=None,
                        avg_num_neighbors: float = None,
                        mode: str = "energy",
                        per_particle: bool = False,
                        positive_species: bool = False,
                        **ace_kwargs
                        ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                              md_util.Array]]:
    """ACE model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of ACE and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "energy" (default),
            returns the total energy of the system. If "property_prediction", 
            returns the many-body basis functions.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        ace_kwargs: Kwargs to change the default structure of ACE.
            For definition of the kwargs, see ACE.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[ACE] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[ACE] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = functools.partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < position.shape[0],
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(e3nn.Irreps("1o"), vectors)

        net = ACE(
            num_species=n_species,
            rcut=r_cutoff,
            **ace_kwargs
        )

        per_atom_energies, features = net(
            vectors, senders, receivers, species, mask)

        if mode == "energy":
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)
