import argparse
import ast
import os
import sys

# Ensure CUDA_VISIBLE_DEVICES is set BEFORE importing JAX or cgbench (which imports JAX)
if "--device" in sys.argv:
    _idx = sys.argv.index("--device")
    if _idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_idx + 1]
elif "GPU_CHOICE" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_CHOICE"]

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")

# Add parent directory to path to import cgbench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from cgbench.core.mapping import register_custom_map


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--cgmap", type=str, help="CG mapping to use", default=None)
parser.add_argument(
    "--custom-cg-map",
    type=str,
    help="Custom fixed CG mapping array as string, e.g., '[8, 8, 2, 4, 7, 0, ...]'",
    default=None,
)
parser.add_argument("--mol", type=str, help="Molecule to use", required=True)
parser.add_argument("--prior", action="store_true", help="Use bond priors")
parser.add_argument(
    "--rcut", type=float, help="Cutoff radius for neighbor list", default=0.5
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output", default=False
)
parser.add_argument(
    "--use-so3",
    action="store_true",
    help="Use SO(3) equivariance in MACE instead of O(3) (disables cueq)",
)
parser.add_argument(
    "--freeze-cg",
    action="store_true",
    help="Freeze learned CG mapping weights after initialization",
)
parser.add_argument(
    "--freeze-cg-after-epoch",
    type=int,
    help="Freeze learned CG mapping weights after a specific epoch number (0-indexed)",
    default=None,
)
parser.add_argument(
    "--gumbel-temp",
    type=str,
    help="Gumbel temperature configuration: a number for constant temp or a string for schedule ('exponential' or 'linear')",
    default=None,
)
parser.add_argument(
    "--gumbel-temp-min",
    type=float,
    help="Minimum temperature for Gumbel schedule",
    default=None,
)
parser.add_argument(
    "--gumbel-temp-max",
    type=float,
    help="Maximum/starting temperature for Gumbel schedule",
    default=None,
)
parser.add_argument(
    "--gumbel-decay-rate",
    type=float,
    help="Decay rate multiplier per epoch for exponential Gumbel schedule (overrides auto decay)",
    default=None,
)
parser.add_argument(
    "--epochs",
    type=int,
    help="Number of epochs to train",
    default=None,
)
parser.add_argument(
    "--no-direct-force-mapping",
    action="store_false",
    dest="use_direct_force_mapping",
    default=True,
    help="Disable direct force mapping and use coordinate map weights instead",
)
parser.add_argument(
    "--gumbel-temp-3phase-points",
    type=float,
    nargs=4,
    help="4 temperature points for 3phase schedule",
    default=None,
)
parser.add_argument(
    "--gumbel-temp-3phase-timings",
    type=float,
    nargs=2,
    help="2 timing fractions for middle points in 3phase schedule",
    default=None,
)
parser.add_argument(
    "--label",
    type=str,
    help="Optional label to append to training output results directory name",
    default=None,
)
parser.add_argument(
    "--unique-cg-species",
    action="store_true",
    help="Use a unique species for each coarse-grained bead instead of sharing species from the heuristic map.",
)
parser.add_argument(
    "--learned-species-embedding",
    action="store_true",
    help="Learn a feature vector for each atom type and combine (add) them to get species features for MACE.",
)
parser.add_argument(
    "--species-embedding-dim",
    type=int,
    help="Dimension of learned species embedding features for MACE",
    default=None,
)
parser.add_argument(
    "--normalize-atom-embedding",
    action="store_true",
    help="L2-normalize learned atom type embeddings before combining them into CG bead features for MACE.",
)
parser.add_argument(
    "--cg-init",
    type=str,
    choices=["random", "hmerged"],
    default=None,
    help="Initialization strategy for learned CG mapping weights ('random' [default] or 'hmerged' [heuristic map]).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="PRNG key seed for random initialization and dataset shuffling.",
)
args = parser.parse_args()

device_choice = args.device or os.environ.get("GPU_CHOICE")
if device_choice and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = device_choice

import jax
import numpy as onp
import optax
from trainer import CGForceMatching
from chemtrain.data import preprocessing
from chemtrain.compose import mace_jax as mace_jax_compose
import json
from jax import numpy as jnp, tree_util
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from cgbench.core import dataset, mapping
from cgbench.core.config import DEFAULT_MACE_CONFIG as MACE_CONFIG, DEFAULT_TRAIN_CONFIG as TRAIN_CONFIG, BOND_SPRING_CONSTANTS
from jax_md import space, energy, partition

from model import GumbelCGAssignment

cg_map_choice = args.cgmap
custom_cg_array = None

if args.custom_cg_map is not None:
    custom_cg_str = args.custom_cg_map.strip()
    try:
        custom_cg_array = ast.literal_eval(custom_cg_str)
        if not isinstance(custom_cg_array, (list, tuple)):
            raise ValueError("Custom CG map must be a list or tuple of integers.")
        custom_cg_array = [int(x) for x in custom_cg_array]
    except Exception as e:
        raise ValueError(f"Failed to parse --custom-cg-map string '{args.custom_cg_map}': {e}")
    if cg_map_choice is None:
        cg_map_choice = "custom"
elif cg_map_choice is not None and cg_map_choice.strip().startswith("["):
    custom_cg_str = cg_map_choice.strip()
    try:
        custom_cg_array = ast.literal_eval(custom_cg_str)
        if not isinstance(custom_cg_array, (list, tuple)):
            raise ValueError("Custom CG map must be a list or tuple of integers.")
        custom_cg_array = [int(x) for x in custom_cg_array]
    except Exception as e:
        raise ValueError(f"Failed to parse --cgmap array string '{cg_map_choice}': {e}")
    cg_map_choice = "custom"

if cg_map_choice is None:
    cg_map_choice = MACE_CONFIG.get("CG_map", "learned")

if cg_map_choice == "custom" and custom_cg_array is None:
    raise ValueError("CG_map is set to 'custom', but no custom mapping array was provided via --custom-cg-map or --cgmap.")

MACE_CONFIG["r_cutoff"] = args.rcut
MACE_CONFIG["mol"] = args.mol 
MACE_CONFIG["CG_map"] = cg_map_choice
MACE_CONFIG["use_bond_priors"] = args.prior
MACE_CONFIG["type"] = "CG" if MACE_CONFIG["CG_map"] != "AT" else "AT"
MACE_CONFIG["freeze_cg"] = args.freeze_cg or MACE_CONFIG.get("freeze_cg", False)
if args.freeze_cg_after_epoch is not None:
    MACE_CONFIG["freeze_cg_after_epoch"] = args.freeze_cg_after_epoch
    TRAIN_CONFIG["freeze_cg_after_epoch"] = args.freeze_cg_after_epoch

MACE_CONFIG["use_direct_force_mapping"] = args.use_direct_force_mapping
MACE_CONFIG["unique_cg_species"] = args.unique_cg_species or MACE_CONFIG.get("unique_cg_species", False)
MACE_CONFIG["learned_species_embedding"] = args.learned_species_embedding or MACE_CONFIG.get("learned_species_embedding", False)
if MACE_CONFIG["unique_cg_species"] and MACE_CONFIG["learned_species_embedding"]:
    raise ValueError(
        "[ERROR] --unique-cg-species and --learned-species-embedding are mutually exclusive and cannot both be activated."
    )

if args.species_embedding_dim is not None:
    MACE_CONFIG["species_embedding_dim"] = args.species_embedding_dim

MACE_CONFIG["normalize_atom_embedding"] = args.normalize_atom_embedding or MACE_CONFIG.get("normalize_atom_embedding", False)

if args.cg_init is not None:
    MACE_CONFIG["cg_init"] = args.cg_init
else:
    MACE_CONFIG["cg_init"] = MACE_CONFIG.get("cg_init", "random")

if args.seed is not None:
    MACE_CONFIG["PRNGKey_seed"] = args.seed

gumbel_temp_choice = args.gumbel_temp
if gumbel_temp_choice is None:
    gumbel_temp_choice = TRAIN_CONFIG.get("gumbel_temp", "exponential")
TRAIN_CONFIG["gumbel_temp"] = gumbel_temp_choice

if args.gumbel_temp_min is not None:
    TRAIN_CONFIG["gumbel_temp_min"] = args.gumbel_temp_min
if args.gumbel_temp_max is not None:
    TRAIN_CONFIG["gumbel_temp_max"] = args.gumbel_temp_max
if args.gumbel_decay_rate is not None:
    TRAIN_CONFIG["gumbel_decay_rate"] = args.gumbel_decay_rate

if args.gumbel_temp_3phase_points is not None:
    TRAIN_CONFIG["gumbel_temp_3phase_points"] = args.gumbel_temp_3phase_points
if args.gumbel_temp_3phase_timings is not None:
    TRAIN_CONFIG["gumbel_temp_3phase_timings"] = args.gumbel_temp_3phase_timings

if args.epochs is not None:
    TRAIN_CONFIG["num_epochs"] = args.epochs

freeze_after = MACE_CONFIG.get("freeze_cg_after_epoch", None)
if (MACE_CONFIG["freeze_cg"] or freeze_after is not None) and MACE_CONFIG["CG_map"] != "learned":
    print("[WARNING] --freeze-cg or --freeze-cg-after-epoch was specified, but CG mapping choice is not 'learned'. Freezing has no effect.")


# -------------------------
# Load dataset
# -------------------------
if MACE_CONFIG["mol"] == "ala2":
    data = dataset.Ala2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
elif MACE_CONFIG["mol"] == "ala15":
    data = dataset.Ala15_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
elif MACE_CONFIG["mol"] == "hexane":
    data = dataset.Hexane_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"],
        val_ratio=MACE_CONFIG["val_ratio"],
    )
elif MACE_CONFIG["mol"] == "pro2":
    data = dataset.Pro2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
elif MACE_CONFIG["mol"] == "thr2":
    data = dataset.Thr2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
elif MACE_CONFIG["mol"] == "gly2":
    data = dataset.Gly2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
else:
    raise ValueError(
        "Invalid molecule. Use 'ala2', 'ala15', 'hexane', 'pro2', 'thr2', or 'gly2'."
    )
    
# We load the atomistic dataset for AT and learned coarse-graining.
# For static coarse-graining, we coarse-grain the dataset and load cg_dataset_U.
if custom_cg_array is not None and hasattr(data, "map_obj"):
    register_custom_map(data.map_obj, custom_cg_array)

if MACE_CONFIG["type"] == "CG" and MACE_CONFIG["CG_map"] != "learned":
    data.coarse_grain(map=MACE_CONFIG["CG_map"])
    dataset = data.cg_dataset_U
    species = data.cg_species
    masses = data.cg_masses
    # We map species to a 1-indexed contiguous range [1..n_species].
    # WHY: chemtrain's AtomicNumberMapping uses lookup_table[species - 1] to convert input species
    # into internal 0-based MACE embeddings [0..n_species - 1].
    # By ensuring input species are 1-indexed (1..N), species k maps cleanly to internal index k-1,
    # avoiding 0-index underflow (0 - 1 = -1 -> index 99) and out-of-bounds errors when n_species >= 6.
    _, species_dense = onp.unique(species, return_inverse=True)
    species = (species_dense + 1).astype(onp.int32)
    n_species = len(set(species.tolist()))
    print(f"[INFO] Static CG mode: {n_species} unique species mapped to 1-based contiguous range [1..{n_species}].")
else:
    dataset = data.dataset_U
    species = data.species
    masses = data.masses
    # Map species to a 1-indexed contiguous range [1..n_species] for AtomicNumberMapping compatibility.
    _, species_dense = onp.unique(species, return_inverse=True)
    species = (species_dense + 1).astype(onp.int32)
    n_species = len(set(species.tolist()))
    print(f"[INFO] Atomistic mode: {n_species} unique species mapped to 1-based contiguous range [1..{n_species}].")

if MACE_CONFIG["CG_map"] == "learned":
    freeze_after = MACE_CONFIG.get("freeze_cg_after_epoch", None)
    if MACE_CONFIG["freeze_cg"]:
        freeze_suffix = f"_freezecg={MACE_CONFIG['freeze_cg']}"
    elif freeze_after is not None:
        freeze_suffix = f"_freezecg_after={freeze_after}"
    else:
        freeze_suffix = ""
    gumbel_temp_suffix = f"_gumbeltemp={TRAIN_CONFIG['gumbel_temp']}"
    if not MACE_CONFIG["use_direct_force_mapping"]:
        direct_force_map_suffix = "_directforcemap=False"
    else:
        direct_force_map_suffix = ""
    if MACE_CONFIG["unique_cg_species"]:
        species_suffix = "_uniquespecies=True"
    elif MACE_CONFIG["learned_species_embedding"]:
        norm_suffix = "_normatom=True" if MACE_CONFIG.get("normalize_atom_embedding", False) else ""
        species_suffix = f"_learnedspecies=True{norm_suffix}"
    else:
        species_suffix = ""
    if MACE_CONFIG.get("cg_init") != "random":
        cg_init_suffix = f"_cginit={MACE_CONFIG.get('cg_init')}"
    else:
        cg_init_suffix = ""
    print("[INFO] Gumbel temperature: ", TRAIN_CONFIG["gumbel_temp"])
else:
    freeze_suffix = ""
    gumbel_temp_suffix = ""
    direct_force_map_suffix = ""
    species_suffix = ""
    cg_init_suffix = ""

label_suffix = f"_{args.label}" if args.label is not None else ""
output_dir = f"outputs/MLP_train/{MACE_CONFIG['mol'].capitalize()}_map={MACE_CONFIG['CG_map']}{freeze_suffix}{gumbel_temp_suffix}{direct_force_map_suffix}{species_suffix}{cg_init_suffix}_tr={MACE_CONFIG['train_ratio']}_rcut={MACE_CONFIG['r_cutoff']}_epochs={TRAIN_CONFIG['num_epochs']}_int={MACE_CONFIG['num_interactions']}{label_suffix}"
os.makedirs(output_dir, exist_ok=True)

if MACE_CONFIG["CG_map"] == "learned":
    try:
        from cgbench.plotting.training import plot_temperature_schedule
        plot_temperature_schedule(
            gumbel_temp_choice=TRAIN_CONFIG["gumbel_temp"],
            epochs=TRAIN_CONFIG["num_epochs"],
            out_dir=output_dir,
            t_min=TRAIN_CONFIG["gumbel_temp_min"],
            t_max=TRAIN_CONFIG["gumbel_temp_max"],
            decay_rate=TRAIN_CONFIG.get("gumbel_decay_rate"),
            threephase_points=TRAIN_CONFIG.get("gumbel_temp_3phase_points"),
            threephase_timings=TRAIN_CONFIG.get("gumbel_temp_3phase_timings"),
        )
        print(f"[INFO] Saved temperature schedule plot to {output_dir}/gumbel_temperature_schedule.png")
    except Exception as e:
        print(f"[WARNING] Could not plot temperature schedule: {e}")


# -------------------------
# Setup neighbor list and MACE model
# -------------------------
box = data.box
displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

# Lookup target map and num_cg_beads
num_cg_beads = 10
initial_mapping = None
cg_species = onp.zeros(num_cg_beads, dtype=onp.int32)

if MACE_CONFIG["type"] == "CG":
    target_map_name = MACE_CONFIG["CG_map"]
    if target_map_name == "learned":
        target_map_name = "custom" if custom_cg_array is not None else "hmerged"  # Reference map to derive bead count and initial species for learned CG mapping

    mol_name_map = {
        "ala2": "Ala2_Map",
        "ala15": "Ala15_Map",
        "hexane": "Hexane_Map",
        "pro2": "Pro2_Map",
        "thr2": "Thr2_Map",
        "gly2": "Gly2_Map",
    }

    if args.mol in mol_name_map:
        class_name = mol_name_map[args.mol]
        map_class = getattr(mapping, class_name, None)
        if map_class is not None:
            try:
                map_inst = getattr(data, "map_obj", map_class())
                if custom_cg_array is not None and hasattr(map_inst, "_maps"):
                    register_custom_map(map_inst, custom_cg_array)

                # If target_map_name is not in the maps, look for a 10-bead map
                if target_map_name not in map_inst.get_available_maps():
                    for map_name in map_inst.get_available_maps():
                        indices, cg_species_tmp, _, _ = map_inst.get_map(map_name)
                        if len(cg_species_tmp) == 10:
                            target_map_name = map_name
                            break
                
                if target_map_name in map_inst.get_available_maps():
                    indices, cg_species, _, _ = map_inst.get_map(target_map_name)
                    num_cg_beads = len(cg_species)
                    if MACE_CONFIG.get("unique_cg_species", False):
                        cg_species = onp.arange(1, num_cg_beads + 1, dtype=onp.int32)
                    indices_clean = [idx if idx >= 0 else 0 for idx in indices]
                    initial_mapping = tuple(indices_clean)
                    if MACE_CONFIG["CG_map"] == "learned":
                        if MACE_CONFIG.get("cg_init") == "random":
                            print(f"[INFO] Initializing learned CG mapping using RANDOM weights (bead count & species derived from '{target_map_name}' map)")
                        else:
                            print(f"[INFO] Initializing learned CG mapping using heuristic '{target_map_name}' map: {initial_mapping}")
                    else:
                        print(f"[INFO] Using static '{target_map_name}' map: {initial_mapping}")
                    print(f"[INFO] Number of CG beads: {num_cg_beads}")
                    print(f"[INFO] Using CG species: {cg_species}")
                    MACE_CONFIG["cg_species"] = cg_species.tolist()
                else:
                    print(f"[WARNING] No map found. Using default 10 beads. CG species will default to all zeros.")
                    if MACE_CONFIG.get("unique_cg_species", False):
                        cg_species = onp.arange(1, num_cg_beads + 1, dtype=onp.int32)
                    MACE_CONFIG["cg_species"] = cg_species.tolist()
            except Exception as e:
                print(f"[WARNING] Error reading heuristic map for initialization: {e}")

if MACE_CONFIG["CG_map"] == "learned":
    if MACE_CONFIG["learned_species_embedding"]:
        n_species = MACE_CONFIG.get("species_embedding_dim", 16)
        print(f"[INFO] MACE model initialized with learned species embedding dim={n_species}")
    else:
        # Map species to a 1-indexed contiguous range [1..n_species] for AtomicNumberMapping compatibility.
        _, cg_species_dense = onp.unique(cg_species, return_inverse=True)
        cg_species = (cg_species_dense + 1).astype(onp.int32)
        n_species = len(set(cg_species.tolist()))
        print(f"[INFO] Learned CG mode: {n_species} unique species mapped to 1-based contiguous range [1..{n_species}].")

    # Create mass-weighted mapping matrix for neighbor list allocation
    # Must match the static path's get_map_weights (m_i / M_I) to ensure
    # identical avg_num_neighbors and edge capacity for MACE initialization.
    initial_map_arr = jnp.array(initial_mapping, dtype=jnp.int32)
    at_masses_arr = jnp.array(masses, dtype=jnp.float32)
    cg_masses_arr = jax.ops.segment_sum(at_masses_arr, initial_map_arr, num_cg_beads)
    c_map_init = mapping.get_map_weights(initial_map_arr, at_masses_arr, cg_masses_arr)

    box_tensor = box
    if box_tensor.ndim != 2:
        box_tensor = onp.eye(box_tensor.shape[0]) * box_tensor

    # Convert initial fractional training coordinates to Cartesian coordinates
    R_at_cart = onp.dot(dataset["training"]["R"], box_tensor.T)

    displacement_fn_cart, shift_fn_cart = space.periodic_general(
        box=box_tensor, fractional_coordinates=False
    )

    # Map initial coordinates in Cartesian space
    cg_positions_init_cart, _ = mapping.map_dataset(
        R_at_cart,
        displacement_fn_cart,
        shift_fn_cart,
        c_map_init,
        d_map=c_map_init,
        force_dataset=jnp.zeros_like(R_at_cart)
    )

    # Convert back to fractional coordinates for neighbor list allocation
    inv_box_tensor = onp.linalg.inv(box_tensor)
    cg_positions_init = onp.dot(cg_positions_init_cart, inv_box_tensor.T)

    # Allocate neighbor list nbrs_init for CG beads
    cg_dataset_init = {
        "R": cg_positions_init,
        "box": dataset["training"]["box"],
        "mask": jnp.ones((cg_positions_init.shape[0], num_cg_beads), dtype=jnp.bool_)
    }

    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
        preprocessing.allocate_neighborlist(
            cg_dataset_init,
            displacement_fn,
            box,
            r_cutoff=MACE_CONFIG["r_cutoff"],
            mask_key="mask",
            box_key="box",
            format=partition.Dense,
            batch_size=100,
        )
    )
else:
    # For AT or Static CG training, allocate neighbor list directly on dataset["training"]
    nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
        preprocessing.allocate_neighborlist(
            dataset["training"],
            displacement_fn,
            box,
            r_cutoff=MACE_CONFIG["r_cutoff"],
            mask_key="mask",
            box_key="box",
            format=partition.Dense,
            batch_size=100,
        )
    )

if args.verbose:
    print(
        f"Max neighbors: {max_neighbors}, Max edges: {max_edges}, Avg neighbors: {avg_num_neighbors}"
    )

mace_cfg = {
    # 1-indexed atomic numbers matching 1..n_species so chemtrain's AtomicNumberMapping
    # maps species k (1..N) to model embedding index k - 1 (0..N-1).
    "atomic_numbers": onp.arange(1, n_species + 1, dtype=onp.int32),
    "r_cutoff": MACE_CONFIG["r_cutoff"],
    "hidden_irreps": MACE_CONFIG["hidden_irreps"],
    "MLP_irreps": MACE_CONFIG["readout_mlp_irreps"],
    "num_interactions": MACE_CONFIG["num_interactions"],
    "max_ell": MACE_CONFIG["max_ell"],
    "correlation": MACE_CONFIG["correlation"],
    "n_radial_basis": MACE_CONFIG["n_radial_basis"],
    "output_irreps": MACE_CONFIG["output_irreps"],
    "use_so3": bool(args.use_so3),
}

cueq_config = CuEquivarianceConfig(
    enabled=True,
    layout=("mul_ir"),
    group=("O3"),
    optimize_all=True,
    conv_fusion=True,
)
if args.use_so3:
    print("[NOTE] Using SO(3) equivariance (no CuEquivariance support)")
    cueq_config = None

template_vars, gnn_energy_fn, model_config = mace_jax_compose.mace_jax_neighborlist(
    displacement=displacement_fn,
    r_cutoff=MACE_CONFIG["r_cutoff"],
    n_species=n_species,
    per_particle=False,
    avg_num_neighbors=avg_num_neighbors,
    mode="energy",
    use_custom_batch_fn=True,
    mace_config=mace_cfg,
    cueq_config=cueq_config,
)
"""
template_vars: A dictionary-like object (Flax FrozenDict) containing the initialized model variables.
It primarily contains the 'params' key, which holds the trainable weights of the MACE-JAX model.
"""

init_params = template_vars["params"]
variables = template_vars
if MACE_CONFIG["CG_map"] == "learned":
    species_init = jnp.asarray(cg_species, dtype=jnp.int32)
else:
    species_init = jnp.asarray(species, dtype=jnp.int32)

def energy_fn_template(energy_params):
    vars = {**variables}
    vars["params"] = energy_params

    def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
        del mode
        dynamic_kwargs.setdefault("species", species_init)
        dynamic_kwargs.setdefault("box", box)
        mask = dynamic_kwargs.pop("mask", jnp.ones(pos.shape[0], dtype=jnp.bool_))

        pots = gnn_energy_fn(vars, pos, neighbor, **dynamic_kwargs)
        if pots.ndim == 2 and pots.shape[-1] == 1:
            pots = pots.squeeze(-1)

        if jnp.issubdtype(dynamic_kwargs["species"].dtype, jnp.floating):
            pots = pots * mask
        else:
            atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
            atomic_energies = jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)
            mapped_species = jnp.argmax(dynamic_kwargs["species"][:, None] == atomic_numbers[None, :], axis=-1)
            pots = (pots - atomic_energies[mapped_species]) * mask
        return jnp.sum(pots)
    
    if args.prior:
        key = f"mol={MACE_CONFIG['mol']}_map={MACE_CONFIG['CG_map']}"
        if key not in BOND_SPRING_CONSTANTS:
            fallback_key = f"mol={MACE_CONFIG['mol']}_map=hmerged"
            if fallback_key in BOND_SPRING_CONSTANTS:
                key = fallback_key
                print(f"[WARNING] Prior constants for 'mol={MACE_CONFIG['mol']}_map={MACE_CONFIG['CG_map']}' not found. Falling back to '{fallback_key}'.")
        assert key in BOND_SPRING_CONSTANTS, f"Prior constants for '{key}' not found in BOND_SPRING_CONSTANTS."
        prior_constants = BOND_SPRING_CONSTANTS[key]
                
        harmonic_energy_fn = energy.simple_spring_bond(
            displacement_fn, 
            bond=jnp.asarray(prior_constants['indices']),
            length=jnp.exp(prior_constants['log_b0']), # b0
            epsilon=jnp.exp(prior_constants['log_kb']), # kb
            alpha=2.0 # standard harmonic
        )
        
        def total_energy_fn(pos, neighbor, **dynamic_kwargs):
            gnn_e = energy_fn(pos, neighbor, **dynamic_kwargs)
            harmonic_e = harmonic_energy_fn(pos)
            return gnn_e + harmonic_e
            
        return total_energy_fn
        
    else:
        return energy_fn

r_init = jnp.asarray(dataset["training"]["R"][0])
if MACE_CONFIG["CG_map"] == "learned":
    r_init_cg = cg_positions_init[0]
    mask_init_cg = jnp.ones(num_cg_beads, dtype=jnp.bool_)
    nbrs_init = nbrs_init.update(r_init_cg, mask=mask_init_cg)

    unique_atom_species_tuple = tuple(sorted(list(set(dataset["training"]["species"][0].tolist()))))
    model_initial_mapping = None if MACE_CONFIG.get("cg_init") == "random" else initial_mapping
    cg_model = GumbelCGAssignment(
        num_cg_beads=num_cg_beads,
        initial_mapping=model_initial_mapping,
        learned_species_embedding=MACE_CONFIG["learned_species_embedding"],
        normalize_atom_embedding=MACE_CONFIG.get("normalize_atom_embedding", False),
        species_embedding_dim=MACE_CONFIG["species_embedding_dim"],
        unique_atom_species=unique_atom_species_tuple,
    )
    prng_seed = MACE_CONFIG.get("PRNGKey_seed", 42)
    key_init, key_gumbel = jax.random.split(jax.random.PRNGKey(prng_seed))
    cg_params = cg_model.init(
        key_init,
        r_init,
        dataset["training"]["species"][0],
        key_gumbel,
        atom_masses=masses,
        deterministic=False
    )
else:
    mask_init = jnp.asarray(dataset["training"]["mask"][0])
    nbrs_init = nbrs_init.update(r_init, mask=mask_init)

# -------------------------
# Setup optimizer
# -------------------------
batch_size = TRAIN_CONFIG["batch_size"]
num_samples = dataset["training"]["R"].shape[0]
epochs = TRAIN_CONFIG["num_epochs"]
total_steps = (epochs * num_samples) // batch_size
transition_steps = total_steps

scheduler = optax.exponential_decay(
    init_value=TRAIN_CONFIG["init_lr"],
    transition_steps=transition_steps,
    decay_rate=TRAIN_CONFIG["decay_rate"],
)

optimizer_fm = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0),
)
if args.verbose:
    print(f"Total steps: {total_steps}")
    print(f"Training on {num_samples} samples.")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}")

# -------------------------
# Setup trainer
# -------------------------
from chemtrain.trainers import ForceMatching

def make_cg_labels(params, learned_species_embedding=False):
    def label_fn(path, leaf):
        path_strs = [str(p.key) if hasattr(p, 'key') else str(p) for p in path]
        if path_strs[0] == 'cg_map':
            if 'atom_type_embeddings' in path_strs and learned_species_embedding:
                return 'trainable'
            return 'frozen'
        return 'trainable'
    return jax.tree_util.tree_map_with_path(label_fn, params)

if MACE_CONFIG["CG_map"] == "learned":
    cg_save_freq = TRAIN_CONFIG.get("cg_save_freq", 5)
    joint_params = {'mace': init_params, 'cg_map': cg_params['params']}
    
    if MACE_CONFIG["freeze_cg"]:
        print("[INFO] Freezing learned CG assignment weights.")
        labels = make_cg_labels(joint_params, learned_species_embedding=MACE_CONFIG["learned_species_embedding"])
        optimizer_fm = optax.multi_transform(
            {'trainable': optimizer_fm, 'frozen': optax.set_to_zero()},
            labels
        )

    trainer_fm = CGForceMatching(
        joint_params,
        optimizer_fm,
        energy_fn_template,
        nbrs_init,
        log_file=f"{output_dir}/force_matching.log",
        batch_per_device=int(batch_size),
        model_cg_map=cg_model,
        cg_species=cg_species,
        atom_masses=masses,
        freeze_cg=MACE_CONFIG["freeze_cg"],
        empty_bead_penalty_weight=TRAIN_CONFIG.get("empty_bead_penalty_weight", 0.0),
        use_direct_force_mapping=MACE_CONFIG["use_direct_force_mapping"],
        learned_species_embedding=MACE_CONFIG["learned_species_embedding"]
    )
    trainer_fm.cg_save_freq = cg_save_freq
    trainer_fm.saved_cg_maps = {}
    trainer_fm.saved_cg_gradients = {}

    def check_freeze_cg_after_epoch(trainer, *args, **kwargs):
        epoch = trainer._epoch
        freeze_after = MACE_CONFIG.get("freeze_cg_after_epoch", None)
        if freeze_after is not None and epoch >= freeze_after:
            if not trainer.freeze_cg:
                print(f"[INFO] Epoch {epoch} >= freeze_cg_after_epoch ({freeze_after}): Freezing learned CG assignment weights.")
                trainer.freeze_cg = True

    def print_learned_mapping(trainer, *args, **kwargs):
        cg_map_params = trainer.params.get('cg_map', None)
        if cg_map_params is not None:
            epoch = trainer._epoch

            # Aggregate and save CG gradients every x epochs and at the final epoch
            if hasattr(trainer, "epoch_cg_gradients") and trainer.epoch_cg_gradients:
                if not hasattr(trainer, "saved_cg_gradients"):
                    trainer.saved_cg_gradients = {}
                cg_save_freq = getattr(trainer, "cg_save_freq", 5)
                if epoch % cg_save_freq == 0 or epoch == epochs - 1:
                    avg_kernel_grad = onp.mean([g['kernel'] for g in trainer.epoch_cg_gradients], axis=0)
                    avg_bias_grad = onp.mean([g['bias'] for g in trainer.epoch_cg_gradients], axis=0)
                    trainer.saved_cg_gradients[int(epoch)] = {
                        'kernel': avg_kernel_grad.tolist(),
                        'bias': avg_bias_grad.tolist()
                    }
                trainer.epoch_cg_gradients.clear()

            # Save atom embeddings if learned_species_embedding is enabled
            if MACE_CONFIG["learned_species_embedding"]:
                atom_type_embeddings = cg_map_params.get('atom_type_embeddings', None)
                if atom_type_embeddings is not None:
                    if not hasattr(trainer, "saved_atom_embeddings"):
                        trainer.saved_atom_embeddings = {}
                    cg_save_freq = getattr(trainer, "cg_save_freq", 5)
                    if epoch % cg_save_freq == 0 or epoch == epochs - 1:
                        unique_species = getattr(cg_model, "unique_atom_species", None)
                        emb_arr = jax.device_get(atom_type_embeddings)
                        emb_dict = {}
                        if unique_species is not None:
                            for idx, spec in enumerate(unique_species):
                                emb_dict[int(spec)] = emb_arr[idx].tolist()
                        else:
                            for idx in range(emb_arr.shape[0]):
                                emb_dict[int(idx)] = emb_arr[idx].tolist()
                        trainer.saved_atom_embeddings[int(epoch)] = emb_dict

            dense_keys = [k for k in cg_map_params.keys() if 'Dense' in k]
            if dense_keys:
                dense_params = cg_map_params[dense_keys[0]]
                kernel = dense_params.get('kernel', None)
                bias = dense_params.get('bias', None)
                if kernel is not None:
                    if bias is not None:
                        logits = kernel + bias[None, :]
                    else:
                        logits = kernel
                    
                    assignments = jnp.argmax(logits, axis=-1)
                    print(f"\n--- Learned CG Mapping at Epoch {epoch} ---")
                    
                    # Print kernel stats
                    k_min, k_max, k_mean = jnp.min(kernel), jnp.max(kernel), jnp.mean(kernel)
                    print(f"[DEBUG] Mapping weights stats - Min: {k_min:.4f}, Max: {k_max:.4f}, Mean: {k_mean:.4f}")
                    
                    assignments_list = [int(x) for x in jax.device_get(assignments)]
                    print(f"Atom assignments to CG beads: {assignments_list}")
                    
                    # Save CG map every x epochs and at the final epoch
                    if not hasattr(trainer, "saved_cg_maps"):
                        trainer.saved_cg_maps = {}
                    cg_save_freq = getattr(trainer, "cg_save_freq", 5)
                    if epoch % cg_save_freq == 0 or epoch == epochs - 1:
                        trainer.saved_cg_maps[int(epoch)] = assignments_list
                    
                    bead_to_atoms = {}
                    for atom_idx, bead_idx in enumerate(assignments_list):
                        bead_to_atoms.setdefault(bead_idx, []).append(atom_idx)
                    
                    print("CG Beads to Atoms:")
                    for bead_idx in sorted(bead_to_atoms.keys()):
                        print(f"  Bead {bead_idx}: Atoms {bead_to_atoms[bead_idx]}")

                    print("-------------------------------------------\n", flush=True)

    def update_gumbel_temperature(trainer, *args, **kwargs):
        epoch = trainer._epoch
        try:
            gumbel_temp_val = float(gumbel_temp_choice)
            trainer.temperature = gumbel_temp_val
            if epoch == 0:
                print(f"[INFO] Gumbel-Softmax temperature set to constant {trainer.temperature:.4f}")
        except ValueError:
            t_start = TRAIN_CONFIG.get("gumbel_temp_max", 1.0)
            t_min = TRAIN_CONFIG.get("gumbel_temp_min", 0.1)
            if gumbel_temp_choice == "exponential":
                decay_rate = TRAIN_CONFIG.get("gumbel_decay_rate")
                if decay_rate is None:
                    decay_rate = (t_min / t_start) ** (1.0 / (epochs - 1)) if epochs > 1 else 1.0
                trainer.temperature = max(t_min, t_start * (decay_rate ** epoch))
            elif gumbel_temp_choice == "linear":
                trainer.temperature = max(t_min, t_start - (t_start - t_min) * (epoch / (epochs - 1)) if epochs > 1 else 0.0)
            elif gumbel_temp_choice == "3phase":
                t_pts = TRAIN_CONFIG.get("gumbel_temp_3phase_points", [1.0, 0.4, 0.3, 0.1])
                t_tms = TRAIN_CONFIG.get("gumbel_temp_3phase_timings", [0.10, 0.90])
                t0, t1, t2, t3 = t_pts
                f1, f2 = t_tms
                if epochs <= 1:
                    trainer.temperature = t0
                else:
                    e1 = f1 * (epochs - 1)
                    e2 = f2 * (epochs - 1)
                    e3 = epochs - 1
                    if epoch <= e1:
                        trainer.temperature = t0 + (t1 - t0) * (epoch / e1) if e1 > 0 else t1
                    elif epoch <= e2:
                        trainer.temperature = t1 + (t2 - t1) * ((epoch - e1) / (e2 - e1)) if e2 > e1 else t2
                    else:
                        trainer.temperature = t2 + (t3 - t2) * ((epoch - e2) / (e3 - e2)) if e3 > e2 else t3
            else:
                raise ValueError(f"Unknown Gumbel temperature schedule: {gumbel_temp_choice}")
            print(f"[INFO] Gumbel-Softmax temperature set to {trainer.temperature:.4f} for Epoch {epoch}")

    trainer_fm.add_task("pre_epoch", check_freeze_cg_after_epoch)
    trainer_fm.add_task("pre_epoch", update_gumbel_temperature)
    trainer_fm.add_task("post_epoch", print_learned_mapping)
else:
    # Standard Force Matching for AT or static CG
    trainer_fm = ForceMatching(
        init_params,
        optimizer_fm,
        energy_fn_template,
        nbrs_init,
        log_file=f"{output_dir}/force_matching.log",
        batch_per_device=int(batch_size),
    )

trainer_fm.set_dataset(dataset["training"], stage="training", rng_seed=MACE_CONFIG.get("PRNGKey_seed", 42))
trainer_fm.set_dataset(dataset["validation"], stage="validation", include_all=True)
if "testing" in dataset and dataset["testing"]["R"].shape[0] >= batch_size:
    trainer_fm.set_dataset(dataset["testing"], stage="testing", include_all=True)

from trainer import evaluate_per_bead_forces

trainer_fm.saved_per_bead_losses = {
    "epochs": [],
    "val_mean": [],
    "val_var": [],
}

def track_per_bead_losses(trainer, *args, **kwargs):
    if "validation" in dataset:
        try:
            b_size = int(batch_size) if 'batch_size' in globals() else 32
            mean_err, var_err = evaluate_per_bead_forces(trainer, dataset["validation"], batch_size=b_size)
            trainer.saved_per_bead_losses["epochs"].append(int(trainer._epoch))
            trainer.saved_per_bead_losses["val_mean"].append(mean_err.tolist())
            trainer.saved_per_bead_losses["val_var"].append(var_err.tolist())
        except Exception as e:
            print(f"[WARNING] Could not track per-bead force losses at epoch {trainer._epoch}: {e}")

trainer_fm.add_task("post_epoch", track_per_bead_losses)


# -------------------------
# Run training and save results
# -------------------------
# Train and save the results to a new folder
trainer_fm.train(epochs)
trainer_fm.save_trainer(f"{output_dir}/trainer.pkl", format=".pkl")
trainer_fm.save_energy_params(f"{output_dir}/best_params.pkl", ".pkl", best=True)
trainer_fm.save_energy_params(f"{output_dir}/final_params.pkl", ".pkl", best=False)

# Save configs as json
with open(f"{output_dir}/config.json", "w") as f:
    json.dump(MACE_CONFIG, f, indent=4)
# Save training config as json
with open(f"{output_dir}/train_config.json", "w") as f:
    json.dump(TRAIN_CONFIG, f, indent=4)

if MACE_CONFIG["CG_map"] == "learned":
    saved_cg_maps_dict = getattr(trainer_fm, "saved_cg_maps", {})
    cg_maps_path = f"{output_dir}/cg_maps.json"
    with open(cg_maps_path, "w") as f:
        json.dump(saved_cg_maps_dict, f, indent=4)
    print(f"[INFO] Saved CG maps to {cg_maps_path}")

    if MACE_CONFIG["learned_species_embedding"] and hasattr(trainer_fm, "saved_atom_embeddings") and trainer_fm.saved_atom_embeddings:
        atom_emb_path = f"{output_dir}/atom_embeddings.json"
        with open(atom_emb_path, "w") as f:
            json.dump(trainer_fm.saved_atom_embeddings, f, indent=4)
        print(f"[INFO] Saved atom embeddings to {atom_emb_path}")

        try:
            from cgbench.plotting.training import plot_atom_embeddings_grid
            plot_atom_embeddings_grid(trainer_fm.saved_atom_embeddings, output_dir)
        except Exception as e:
            print(f"[WARNING] Could not generate atom embeddings grid visualization: {e}")

    if hasattr(trainer_fm, "saved_cg_gradients") and trainer_fm.saved_cg_gradients:
        cg_grads_path = f"{output_dir}/cg_gradients.json"
        with open(cg_grads_path, "w") as f:
            json.dump(trainer_fm.saved_cg_gradients, f, indent=4)
        print(f"[INFO] Saved CG gradients to {cg_grads_path}")

        try:
            import subprocess
            viz_script = os.path.join(os.path.dirname(__file__), "visualize_cg_gradients.py")
            subprocess.run([sys.executable, viz_script, cg_grads_path], check=True)
        except Exception as e:
            print(f"[WARNING] Could not automatically generate gradient visualization: {e}")
else:
    # Static or Custom CG map
    static_map_list = list(initial_mapping) if initial_mapping is not None else []
    saved_cg_maps_dict = {"0": static_map_list}
    cg_maps_path = f"{output_dir}/cg_maps.json"
    with open(cg_maps_path, "w") as f:
        json.dump(saved_cg_maps_dict, f, indent=4)
    print(f"[INFO] Saved CG map to {cg_maps_path}")

# Always visualize the CG map (single image for static/custom, grid over time for learned)
try:
    viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../mapping_viz"))
    if viz_dir not in sys.path:
        sys.path.insert(0, viz_dir)
    from viz import visualize_mapping, visualize_mapping_grid, get_molecule_with_node_ordering

    mol_name = MACE_CONFIG["mol"]
    mol = get_molecule_with_node_ordering(mol_name)

    if MACE_CONFIG["CG_map"] == "learned" and hasattr(trainer_fm, "saved_cg_maps") and len(trainer_fm.saved_cg_maps) > 1:
        sorted_epochs = sorted(trainer_fm.saved_cg_maps.keys())
        mappings = [trainer_fm.saved_cg_maps[ep] for ep in sorted_epochs]
        legends = [f"Epoch {ep}" for ep in sorted_epochs]
        epoch_species = [cg_species.tolist() for _ in sorted_epochs]

        output_image_path = f"{output_dir}/cg_maps_over_time.png"
        visualize_mapping_grid(mol, mappings, legends, epoch_species, output_image_path)
        print(f"[INFO] Saved CG maps visualization over time to {output_image_path}")
    else:
        # Single fixed map (static, custom, or single epoch)
        if MACE_CONFIG["CG_map"] == "learned" and hasattr(trainer_fm, "saved_cg_maps") and trainer_fm.saved_cg_maps:
            final_ep = max(trainer_fm.saved_cg_maps.keys())
            map_to_plot = trainer_fm.saved_cg_maps[final_ep]
            legend_str = f"Learned CG Map (Epoch {final_ep})"
        else:
            map_to_plot = list(initial_mapping) if initial_mapping is not None else []
            legend_str = f"CG Map ({MACE_CONFIG['CG_map']})"

        output_image_path = f"{output_dir}/cg_map.png"
        visualize_mapping(mol, map_to_plot, output_image_path, species=cg_species.tolist(), legend=legend_str)
        print(f"[INFO] Saved CG map visualization to {output_image_path}")
except Exception as e:
    print(f"[WARNING] Could not generate CG map visualization: {e}")

from cgbench.plotting.training import plot_predictions, plot_convergence, plot_per_bead_force_losses

# Plot training convergence
plot_convergence(trainer_fm, output_dir)

# Save per-bead force losses json and plot
if hasattr(trainer_fm, "saved_per_bead_losses") and trainer_fm.saved_per_bead_losses["epochs"]:
    per_bead_json_path = f"{output_dir}/per_bead_force_losses.json"
    with open(per_bead_json_path, "w") as f:
        json.dump(trainer_fm.saved_per_bead_losses, f, indent=4)
    print(f"[INFO] Saved per-bead force losses to {per_bead_json_path}")

    try:
        plot_per_bead_force_losses(trainer_fm.saved_per_bead_losses, output_dir)
    except Exception as e:
        print(f"[WARNING] Could not plot per-bead force losses: {e}")


predictions_val = trainer_fm.predict(
    dataset["validation"],
    trainer_fm.best_params,
    batch_size=batch_size,
)
predictions_val = tree_util.tree_map(onp.asarray, predictions_val)
if MACE_CONFIG["CG_map"] == "learned":
    # predictions are at CG resolution; compare against the CG-mapped reference forces
    cg_ref_val = {"F": predictions_val["Mapped_Target_F"]}
    plot_predictions(predictions_val, cg_ref_val, output_dir, name="preds_validation")
else:
    plot_predictions(
        predictions_val, dataset["validation"], output_dir, name="preds_validation"
    )

if "testing" in dataset and dataset["testing"]["R"].shape[0] >= batch_size:
    predictions_test = trainer_fm.predict(
        dataset["testing"],
        trainer_fm.best_params,
        batch_size=batch_size,
    )
    predictions_test = tree_util.tree_map(onp.asarray, predictions_test)
    onp.savez(f"{output_dir}/predictions_test.npz", **predictions_test)
    if MACE_CONFIG["CG_map"] == "learned":
        cg_ref_test = {"F": predictions_test["Mapped_Target_F"]}
        plot_predictions(predictions_test, cg_ref_test, output_dir, name="preds_testing")
    else:
        plot_predictions(
            predictions_test, dataset["testing"], output_dir, name="preds_testing"
        )
