import functools
from typing import Callable, Optional, Tuple, Set, Union, List
import math

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, lax

import numpy as onp


class RadialBasisLayer(hk.Module):
    """Radial basis functions for ACE model.
    
    Implements Rnl(r) = Pn(x) * env(r) where:
    - Pn(x) are orthogonal polynomials (e.g., Chebyshev, Legendre)
    - env(r) is an envelope function for smooth cutoff
    - x is a transformed coordinate x = (r - r0) / (rcut - r0)
    """
    
    def __init__(
        self,
        n_max: int,
        l_max: int,
        rcut: float,
        r0: float = 0.0,
        polynomial_type: str = "chebyshev",
        envelope_type: str = "poly",
        envelope_p: int = 6,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.n_max = n_max
        self.l_max = l_max
        self.rcut = rcut
        self.r0 = r0
        self.polynomial_type = polynomial_type
        self.envelope_type = envelope_type
        self.envelope_p = envelope_p
        
        # Create basis specification
        self.spec = []
        for n in range(n_max):
            for l in range(l_max + 1):
                self.spec.append((n, l))
        
        # Initialize polynomial coefficients
        self._init_polynomials()
    
    def _init_polynomials(self):
        """Initialize polynomial basis functions."""
        if self.polynomial_type == "chebyshev":
            # Chebyshev polynomials of the first kind
            self.polynomials = self._chebyshev_polynomials
        elif self.polynomial_type == "legendre":
            # Legendre polynomials
            self.polynomials = self._legendre_polynomials
        else:
            raise ValueError(f"Unknown polynomial type: {self.polynomial_type}")
    
    def _chebyshev_polynomials(self, x: jnp.ndarray, n: int) -> jnp.ndarray:
        """Evaluate Chebyshev polynomial of degree n at x."""
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        elif n == 2:
            return 2 * x**2 - 1
        elif n == 3:
            return 4 * x**3 - 3 * x
        elif n == 4:
            return 8 * x**4 - 8 * x**2 + 1
        else:
            # For higher degrees, use recurrence relation: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
            T_prev2 = jnp.ones_like(x)  # T_0
            T_prev1 = x  # T_1
            
            for i in range(2, n + 1):
                T_curr = 2 * x * T_prev1 - T_prev2
                T_prev2 = T_prev1
                T_prev1 = T_curr
            
            return T_curr
    
    def _legendre_polynomials(self, x: jnp.ndarray, n: int) -> jnp.ndarray:
        """Evaluate Legendre polynomial of degree n at x."""
        if n == 0:
            return jnp.ones_like(x)
        elif n == 1:
            return x
        elif n == 2:
            return 0.5 * (3 * x**2 - 1)
        elif n == 3:
            return 0.5 * (5 * x**3 - 3 * x)
        elif n == 4:
            return 0.125 * (35 * x**4 - 30 * x**2 + 3)
        else:
            # For higher degrees, use recurrence relation: (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)
            P_prev2 = jnp.ones_like(x)  # P_0
            P_prev1 = x  # P_1
            
            for i in range(2, n + 1):
                P_curr = ((2 * i - 1) * x * P_prev1 - (i - 1) * P_prev2) / i
                P_prev2 = P_prev1
                P_prev1 = P_curr
            
            return P_curr
    
    def envelope(self, r: jnp.ndarray) -> jnp.ndarray:
        """Apply envelope function for smooth cutoff."""
        if self.envelope_type == "poly":
            # Polynomial envelope: (1 - r/rcut)^p for r < rcut, 0 otherwise
            mask = r < self.rcut
            return jnp.where(mask, (1 - r / self.rcut) ** self.envelope_p, 0.0)
        elif self.envelope_type == "exp":
            # Exponential envelope
            mask = r < self.rcut
            return jnp.where(mask, jnp.exp(-r / self.rcut), 0.0)
        else:
            raise ValueError(f"Unknown envelope type: {self.envelope_type}")
    
    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """Evaluate radial basis functions.
        
        Args:
            r: Distances [n_edges]
            
        Returns:
            Rnl: Radial basis values [n_edges, n_max * (l_max + 1)]
        """
        # Apply envelope
        env = self.envelope(r)
        
        # Transform coordinates: r -> x ∈ [-1, 1]
        x = 2 * (r - self.r0) / (self.rcut - self.r0) - 1
        x = jnp.clip(x, -1, 1)  # Ensure x ∈ [-1, 1]
        
        # Evaluate polynomials
        Rnl = []
        for n, l in self.spec:
            Pn = self.polynomials(x, n)
            Rnl.append(Pn * env)
        
        return jnp.stack(Rnl, axis=-1)  # [n_edges, n_max * (l_max + 1)]


class SphericalHarmonicsLayer(hk.Module):
    """Spherical harmonics basis for ACE model."""
    
    def __init__(self, l_max: int, name: Optional[str] = None):
        super().__init__(name=name)
        self.l_max = l_max
        
        # Create specification for (l, m) pairs
        self.spec = []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                self.spec.append((l, m))
    
    def __call__(self, vectors: jnp.ndarray) -> jnp.ndarray:
        """Evaluate spherical harmonics.
        
        Args:
            vectors: Position vectors [n_edges, 3]
            
        Returns:
            Ylm: Spherical harmonics [n_edges, num_harmonics]
        """
        # Convert to spherical coordinates
        r = jnp.linalg.norm(vectors, axis=-1)
        theta = jnp.arccos(vectors[..., 2] / (r + 1e-8))
        phi = jnp.arctan2(vectors[..., 1], vectors[..., 0])
        
        # Evaluate spherical harmonics
        Ylm = []
        for l, m in self.spec:
            Y = self._spherical_harmonic(theta, phi, l, m)
            Ylm.append(Y)
        
        return jnp.stack(Ylm, axis=-1)  # [n_edges, num_harmonics]
    
    def _spherical_harmonic(self, theta: jnp.ndarray, phi: jnp.ndarray, l: int, m: int) -> jnp.ndarray:
        """Evaluate spherical harmonic Y_l^m(theta, phi)."""
        # Simple implementation of spherical harmonics
        # For l=0
        if l == 0:
            return jnp.ones_like(theta) / jnp.sqrt(4 * jnp.pi)
        
        # For l=1
        elif l == 1:
            if m == -1:
                return jnp.sqrt(3 / (4 * jnp.pi)) * jnp.sin(theta) * jnp.sin(phi)
            elif m == 0:
                return jnp.sqrt(3 / (4 * jnp.pi)) * jnp.cos(theta)
            elif m == 1:
                return jnp.sqrt(3 / (4 * jnp.pi)) * jnp.sin(theta) * jnp.cos(phi)
        
        # For l=2
        elif l == 2:
            if m == -2:
                return jnp.sqrt(15 / (16 * jnp.pi)) * jnp.sin(theta)**2 * jnp.sin(2*phi)
            elif m == -1:
                return jnp.sqrt(15 / (4 * jnp.pi)) * jnp.sin(theta) * jnp.cos(theta) * jnp.sin(phi)
            elif m == 0:
                return jnp.sqrt(5 / (16 * jnp.pi)) * (3 * jnp.cos(theta)**2 - 1)
            elif m == 1:
                return jnp.sqrt(15 / (4 * jnp.pi)) * jnp.sin(theta) * jnp.cos(theta) * jnp.cos(phi)
            elif m == 2:
                return jnp.sqrt(15 / (16 * jnp.pi)) * jnp.sin(theta)**2 * jnp.cos(2*phi)
        
        # For higher l, return zeros (simplified)
        return jnp.zeros_like(theta)


class ACETensorProductLayer(hk.Module):
    """Tensor product layer for ACE many-body correlations."""
    
    def __init__(
        self,
        correlation_order: int,
        n_max: int,
        l_max: int,
        num_species: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.correlation_order = correlation_order
        self.n_max = n_max
        self.l_max = l_max
        self.num_species = num_species
        
        # Generate basis specification
        self.basis_spec = self._generate_basis_spec()
        
        # Initialize learnable parameters
        self._init_parameters()
    
    def _generate_basis_spec(self) -> List[Tuple]:
        """Generate basis function specifications for many-body correlations."""
        spec = []
        
        # Generate all possible combinations of (n, l, m) up to correlation_order
        for order in range(1, self.correlation_order + 1):
            # Generate all combinations of basis functions
            # This is a simplified version - in practice, you'd want more sophisticated
            # basis generation that respects rotational invariance
            for n in range(self.n_max):
                for l in range(self.l_max + 1):
                    for m in range(-l, l + 1):
                        spec.append((order, n, l, m))
        
        return spec
    
    def _init_parameters(self):
        """Initialize learnable parameters for tensor products."""
        # Parameters for each species and basis function
        self.W = hk.get_parameter(
            "W",
            shape=(len(self.basis_spec), self.num_species),
            init=hk.initializers.RandomNormal(stddev=0.01)
        )
    
    def __call__(
        self,
        Rnl: jnp.ndarray,  # [n_edges, n_max]
        Ylm: jnp.ndarray,  # [n_edges, num_harmonics]
        node_species: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> jnp.ndarray:
        """Compute tensor products for many-body correlations.
        
        Args:
            Rnl: Radial basis values [n_edges, n_max]
            Ylm: Spherical harmonics [n_edges, num_harmonics]
            node_species: Species of each node
            senders: Sender indices for edges
            receivers: Receiver indices for edges
            
        Returns:
            B: Many-body basis functions [n_nodes, num_basis]
        """
        n_nodes = node_species.shape[0]
        n_edges = Rnl.shape[0]
        
        # Create combined basis functions
        basis_functions = []
        
        # Single-body terms (order 1) - sum radial basis over neighbors
        for n in range(self.n_max):
            R_sum = jax.ops.segment_sum(Rnl[:, n], receivers, n_nodes)
            basis_functions.append(R_sum)
        
        # Two-body terms (order 2) - simplified tensor products
        for n1 in range(min(3, self.n_max)):
            for n2 in range(min(3, self.n_max)):
                if n1 == n2:
                    continue
                
                R1 = Rnl[:, n1]
                R2 = Rnl[:, n2]
                
                # Tensor product
                R_prod = R1 * R2
                R_sum = jax.ops.segment_sum(R_prod, receivers, n_nodes)
                basis_functions.append(R_sum)
        
        # Three-body terms (order 3) - very simplified
        if self.correlation_order >= 3:
            for n1 in range(min(2, self.n_max)):
                for n2 in range(min(2, self.n_max)):
                    for n3 in range(min(2, self.n_max)):
                        if n1 == n2 or n1 == n3 or n2 == n3:
                            continue
                        
                        R1 = Rnl[:, n1]
                        R2 = Rnl[:, n2]
                        R3 = Rnl[:, n3]
                        
                        # Tensor product
                        R_prod = R1 * R2 * R3
                        R_sum = jax.ops.segment_sum(R_prod, receivers, n_nodes)
                        basis_functions.append(R_sum)
        
        # Stack all basis functions
        B = jnp.stack(basis_functions, axis=-1)  # [n_nodes, num_basis]
        
        return B


class ACEReadoutLayer(hk.Module):
    """Readout layer for ACE model."""
    
    def __init__(
        self,
        num_species: int,
        num_basis: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_species = num_species
        self.num_basis = num_basis
        
        # Species-specific linear layer
        self.linear = hk.Linear(1, name="readout")
    
    def __call__(
        self,
        B: jnp.ndarray,  # [n_nodes, num_basis]
        node_species: jnp.ndarray,  # [n_nodes]
    ) -> jnp.ndarray:
        """Compute per-atom energies.
        
        Args:
            B: Many-body basis functions
            node_species: Species of each node
            
        Returns:
            energies: Per-atom energies [n_nodes]
        """
        # Apply species-specific weights
        # For simplicity, we'll use a single linear layer
        # In practice, you'd want species-specific parameters
        
        energies = self.linear(B)  # [n_nodes, 1]
        energies = energies.squeeze(-1)  # [n_nodes]
        
        return energies


class OneBodyLayer(hk.Module):
    """One-body (reference) energy layer."""
    
    def __init__(
        self,
        num_species: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_species = num_species
        
        # Reference energies for each species
        self.E0 = hk.get_parameter(
            "E0",
            shape=(num_species,),
            init=hk.initializers.Constant(0.0)
        )
    
    def __call__(self, node_species: jnp.ndarray) -> jnp.ndarray:
        """Get reference energies for each atom.
        
        Args:
            node_species: Species of each node
            
        Returns:
            E0: Reference energies [n_nodes]
        """
        return self.E0[node_species]


class PairPotentialLayer(hk.Module):
    """Pair potential layer for ACE model."""
    
    def __init__(
        self,
        num_species: int,
        rcut: float,
        n_radial: int = 8,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_species = num_species
        self.rcut = rcut
        self.n_radial = n_radial
        
        # Pair potential parameters
        self.W_pair = hk.get_parameter(
            "W_pair",
            shape=(n_radial, num_species, num_species),
            init=hk.initializers.RandomNormal(stddev=0.01)
        )
        
        # Radial basis for pair potential
        self.radial_basis = RadialBasisLayer(
            n_max=n_radial,
            l_max=0,  # Only l=0 for pair potential
            rcut=rcut,
            name="pair_radial_basis"
        )
    
    def __call__(
        self,
        r: jnp.ndarray,  # [n_edges]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        node_species: jnp.ndarray,  # [n_nodes]
    ) -> jnp.ndarray:
        """Compute pair potential contributions.
        
        Args:
            r: Distances
            senders: Sender indices
            receivers: Receiver indices
            node_species: Species of each node
            
        Returns:
            E_pair: Pair potential energies [n_nodes]
        """
        # Evaluate radial basis
        R_pair = self.radial_basis(r)  # [n_edges, n_radial]
        
        # Get species pairs
        sender_species = node_species[senders]
        receiver_species = node_species[receivers]
        
        # Apply species-specific weights
        E_pair_edges = []
        for i in range(self.n_radial):
            W_ij = self.W_pair[i, sender_species, receiver_species]
            E_pair_edges.append(R_pair[:, i] * W_ij)
        
        E_pair_edges = jnp.stack(E_pair_edges, axis=-1)  # [n_edges, n_radial]
        E_pair_edges = jnp.sum(E_pair_edges, axis=-1)  # [n_edges]
        
        # Sum over neighbors
        n_nodes = node_species.shape[0]
        E_pair = jax.ops.segment_sum(E_pair_edges, receivers, n_nodes)
        
        return E_pair
