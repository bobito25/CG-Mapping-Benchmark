# ACE (Atomic Cluster Expansion) Implementation

This directory contains a JAX/Haiku implementation of the Atomic Cluster Expansion (ACE) model, based on the ACEpotentials.jl library.

## Overview

ACE is a linear model that uses many-body correlations through tensor products of radial basis functions and spherical harmonics. Unlike MACE (Multi-ACE), which uses message passing and non-linear interactions, ACE is a linear model that directly computes many-body basis functions.

## Key Components

### 1. Radial Basis Functions (`RadialBasisLayer`)
- Implements Rnl(r) = Pn(x) * env(r)
- Uses orthogonal polynomials (Chebyshev, Legendre)
- Applies envelope functions for smooth cutoff
- Coordinate transformation: x = (r - r0) / (rcut - r0)

### 2. Spherical Harmonics (`SphericalHarmonicsLayer`)
- Evaluates Ylm(theta, phi) for angular dependence
- Uses e3nn for efficient computation
- Provides rotational invariance

### 3. Tensor Product Layer (`ACETensorProductLayer`)
- Creates many-body correlations through tensor products
- Supports configurable correlation order
- Generates basis functions for different body orders

### 4. Readout Layer (`ACEReadoutLayer`)
- Linear combination of basis functions
- Species-specific parameters
- Outputs per-atom energies

### 5. Optional Components
- **One-body layer**: Reference energies for each species
- **Pair potential**: Two-body interactions

## Usage

### Basic Usage

```python
from model import ace_neighborlist_pp
from jax_md import space, partition

# Create displacement function
displacement_fn = space.free()

# Create ACE model
ace_init, ace_apply = ace_neighborlist_pp(
    displacement=displacement_fn,
    r_cutoff=5.0,
    n_species=2,  # Number of species
    n_max=8,      # Maximum radial basis degree
    l_max=3,      # Maximum angular momentum
    correlation_order=3,  # Many-body correlation order
    include_pair=True,
    include_onebody=True
)

# Initialize parameters
key = jax.random.PRNGKey(42)
params = ace_init(key, positions, nbrs, species)

# Compute energy
energy = ace_apply(params, key, positions, nbrs, species)
```

### Model Parameters

- `num_species`: Number of different atom species
- `n_max`: Maximum radial basis degree (default: 8)
- `l_max`: Maximum angular momentum (default: 3)
- `correlation_order`: Many-body correlation order (default: 3)
- `rcut`: Cutoff radius (default: 5.0)
- `r0`: Inner cutoff (default: 0.0)
- `polynomial_type`: Type of radial polynomials ("chebyshev" or "legendre")
- `envelope_type`: Type of envelope function ("poly" or "exp")
- `envelope_p`: Envelope polynomial order (default: 6)
- `include_pair`: Include pair potential (default: True)
- `include_onebody`: Include one-body reference energy (default: True)

## Comparison with MACE

| Feature | ACE | MACE |
|---------|-----|------|
| Model Type | Linear | Non-linear |
| Architecture | Direct tensor products | Message passing |
| Many-body | Explicit tensor products | Implicit through layers |
| Training | Linear regression | Gradient descent |
| Interpretability | High (linear) | Lower (non-linear) |
| Computational Cost | Lower | Higher |

## Files

- `model.py`: Main ACE model implementation and neighbor list function
- `ace_layers.py`: Core ACE layer implementations
- `layers.py`: MACE layer implementations (for reference)
- `utils.py`: Utility functions
- `test_ace.py`: Test script
- `example_ace.py`: Example usage
- `README_ACE.md`: This documentation

## Testing

Run the test script to verify the implementation:

```bash
python test_ace.py
```

Run the example to see the model in action:

```bash
python example_ace.py
```

## Dependencies

- JAX
- Haiku
- e3nn-jax
- jax-md
- NumPy

## Notes

- This implementation is based on ACEpotentials.jl but adapted for the JAX/Haiku framework
- The tensor product implementation is simplified compared to the full ACE specification
- For production use, consider implementing more sophisticated basis generation
- The model is compatible with the JAX-MD ecosystem for molecular dynamics simulations

## References

- Drautz, R. (2019). Atomic cluster expansion for accurate and transferable interatomic potentials. Physical Review B, 99(1), 014104.
- ACEpotentials.jl: https://github.com/ACEsuit/ACEpotentials.jl
- MACE: https://github.com/ACEsuit/mace
