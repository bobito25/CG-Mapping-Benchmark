"""
Diagnostic tools for detecting and isolating NaNs, Infs, and exploding values
in the coarse-graining training pipeline.
"""
from typing import Dict, Any, Tuple, List, Optional
import jax
import jax.numpy as jnp
import numpy as np


def check_array_jit(name: str, x: jnp.ndarray, extra_info: str = "") -> jnp.ndarray:
    """
    JIT-compatible check that uses jax.debug.print to immediately report if an array
    contains NaNs during forward/loss computation without interrupting execution.
    """
    if not isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
        return x

    def _host_check(val):
        if np.any(np.isnan(val)) or np.any(np.isinf(val)):
            nan_cnt = int(np.sum(np.isnan(val)))
            inf_cnt = int(np.sum(np.isinf(val)))
            print(
                f"[JIT NaN ALERT] Tensor '{name}' {extra_info} -> NaNs: {nan_cnt}/{val.size}, Infs: {inf_cnt}/{val.size}, "
                f"Shape: {val.shape}, Min: {np.nanmin(val)}, Max: {np.nanmax(val)}"
            )

    jax.debug.callback(_host_check, x)
    return x


def check_pairwise_distances(name: str, coords: jnp.ndarray, mask: Optional[jnp.ndarray] = None, min_threshold: float = 1e-4):
    """
    JIT-compatible check to detect overlapping CG beads (distance ~ 0.0) across the batch
    which can cause Hessian and radial basis singularities in GNN potentials.
    coords shape: (batch_size, num_beads, 3) or (num_beads, 3)
    """
    if coords.ndim == 2:
        coords = coords[None, ...]
    
    # coords shape: (batch_size, num_beads, 3)
    # disp: (batch_size, num_beads, num_beads, 3)
    disp = coords[:, :, None, :] - coords[:, None, :, :]
    dist = jnp.sqrt(jnp.sum(disp ** 2, axis=-1) + 1e-12)
    
    num_beads = coords.shape[1]
    eye_mask = jnp.eye(num_beads, dtype=jnp.bool_)[None, ...]  # (1, num_beads, num_beads)
    valid_pairs = ~eye_mask

    if mask is not None:
        if mask.ndim == 1:
            mask = mask[None, ...]
        pair_mask = mask[:, :, None] & mask[:, None, :]
        valid_pairs = valid_pairs & pair_mask

    # Replace non-valid pairs with a large distance
    dist_masked = jnp.where(valid_pairs, dist, 1e6)
    min_dist = jnp.min(dist_masked)
    has_overlap = min_dist < min_threshold

    def _print_overlap():
        jax.debug.print(
            "[JIT OVERLAP ALERT] Active beads in '{name}' are overlapping! Minimum pair distance: {d:.6e} < threshold {th}",
            name=name,
            d=min_dist,
            th=min_threshold,
        )

    jax.lax.cond(has_overlap, _print_overlap, lambda: None)


def inspect_gradients(grad_pytree: Any, step_idx: Optional[int] = None) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Fast host-side inspection of parameter gradients.
    Walks the gradient PyTree and identifies any leaves containing NaNs, Infs, or exploding norms (> 1e8).
    """
    if grad_pytree is None:
        return False, []

    flat_leaves, tree_def = jax.tree_util.tree_flatten_with_path(grad_pytree)
    problematic_layers = []

    for path, leaf in flat_leaves:
        if leaf is None:
            continue
        
        # Format path string e.g. "cg_map/Dense_0/kernel"
        path_str = "/".join(
            str(p.key if hasattr(p, "key") else getattr(p, "idx", str(p)))
            for p in path
        )

        leaf_arr = np.asarray(jax.device_get(leaf))
        if leaf_arr.size == 0:
            continue

        has_nan = bool(np.isnan(leaf_arr).any())
        has_inf = bool(np.isinf(leaf_arr).any())
        
        norm_val = float(np.linalg.norm(leaf_arr)) if not (has_nan or has_inf) else float("nan")
        is_exploding = bool(np.isfinite(norm_val) and norm_val > 1e8)

        if has_nan or has_inf or is_exploding:
            nan_cnt = int(np.sum(np.isnan(leaf_arr)))
            inf_cnt = int(np.sum(np.isinf(leaf_arr)))
            min_val = float(np.nanmin(leaf_arr)) if not np.all(np.isnan(leaf_arr)) else float("nan")
            max_val = float(np.nanmax(leaf_arr)) if not np.all(np.isnan(leaf_arr)) else float("nan")

            problematic_layers.append({
                "path": path_str,
                "shape": leaf_arr.shape,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_exploding": is_exploding,
                "nan_count": nan_cnt,
                "inf_count": inf_cnt,
                "norm": norm_val,
                "min": min_val,
                "max": max_val,
            })

    if problematic_layers:
        has_nan_or_inf = any(item["has_nan"] or item["has_inf"] for item in problematic_layers)
        step_str = f" at step {step_idx}" if step_idx is not None else ""
        alert_title = "NaNs or Infs Detected in Gradients" if has_nan_or_inf else "Large Gradient Norms Detected (Pre-clipping)"
        
        print("\n" + "=" * 80)
        print(f"[DIAGNOSTIC {'ALERT' if has_nan_or_inf else 'INFO'}] {alert_title}{step_str}!")
        print("=" * 80)
        print(f"Total problematic parameter leaves: {len(problematic_layers)}")
        print("-" * 80)
        for item in problematic_layers[:20]:  # limit to top 20 to prevent flood
            status_flags = []
            if item["has_nan"]:
                status_flags.append(f"NaNs: {item['nan_count']}/{np.prod(item['shape'])}")
            if item["inf_count"] > 0:
                status_flags.append(f"Infs: {item['inf_count']}")
            if item["is_exploding"]:
                status_flags.append("LARGE NORM (>1e8)")
            
            flag_str = ", ".join(status_flags)
            print(f"• Layer: {item['path']}")
            print(f"  Shape: {item['shape']} | Status: [{flag_str}]")
            print(f"  Norm: {item['norm']:.4e} | Range: [{item['min']:.4e}, {item['max']:.4e}]")
        
        if len(problematic_layers) > 20:
            print(f"... and {len(problematic_layers) - 20} more layers with issues.")
        print("=" * 80 + "\n", flush=True)
        return has_nan_or_inf, problematic_layers

    return False, []


def inspect_losses(loss: Any, per_target_losses: Optional[Dict[str, Any]] = None, step_idx: Optional[int] = None) -> bool:
    """
    Checks if total loss or any per-target component is NaN or Inf.
    """
    loss_val = float(np.asarray(jax.device_get(loss)))
    has_loss_issue = not np.isfinite(loss_val)

    component_issues = {}
    if per_target_losses is not None:
        for k, v in per_target_losses.items():
            val = float(np.asarray(jax.device_get(v)))
            if not np.isfinite(val):
                component_issues[k] = val

    if has_loss_issue or component_issues:
        step_str = f" at step {step_idx}" if step_idx is not None else ""
        print("\n" + "!" * 80)
        print(f"[DIAGNOSTIC ALERT] NaN or Infinite Loss Detected{step_str}!")
        print(f"Total Training Loss: {loss_val}")
        if component_issues:
            print("Corrupted loss components:")
            for k, val in component_issues.items():
                print(f"  • {k}: {val}")
        print("!" * 80 + "\n", flush=True)
        return True

    return False
