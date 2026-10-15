import argparse
import os
import sys

# Add parent directory to path to import cgbench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Ensure CUDA_VISIBLE_DEVICES is set BEFORE importing JAX or cgbench (which imports JAX)
if "--device" in sys.argv:
    _idx = sys.argv.index("--device")
    if _idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_idx + 1]
elif "GPU_CHOICE" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_CHOICE"]

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--model", type=str, help="Model path", required=True) # Pass path to best_params.pkl
parser.add_argument("--mol", type=str, help="Molecule to simulate", required=True)
parser.add_argument("--t-total", type=float, help="Total simulation time in ps", default=None)
parser.add_argument("--n-chains", type=int, help="Number of parallel chains", default=None)
parser.add_argument("--sim-mode", type=str, help="Simulation mode ('sampling', 'stability', 'speed', etc.)", default=None)
parser.add_argument("--dt-fs", type=float, nargs="+", help="Timestep(s) in fs to simulate", default=None)
parser.add_argument("--ensemble", type=str, choices=["NVT", "NVE"], help="Ensemble (NVT or NVE)", default=None)
parser.add_argument(
    "--cg-map", "--cgmap",
    dest="cg_map",
    type=str,
    help="Manual CG mapping to use (e.g. 'hmerged', 'c-alpha', or a list string like '[0,0,1,1]')",
    default=None,
)
parser.add_argument(
    "--custom-cg-map",
    type=str,
    help="Custom CG mapping as list of bead indices per atom, e.g. '[0,0,1,1]'",
    default=None,
)
parser.add_argument(
    "--map-name",
    type=str,
    help="Optional name/tag for the manual CG map (used in output directory name, e.g. 'co_merged')",
    default=None,
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output", default=True
)
args = parser.parse_args()

import ast
import json
import pickle
import numpy as onp
from jax import numpy as jnp, tree_util
import jax
import time

from chemtrain.data import preprocessing
from chemtrain import quantity, util
from chemtrain.compose import mace_jax as mace_jax_compose
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from jax import random
from chemtrain.ensemble import sampling
from jax_md import partition, space, simulate, energy
from jax_md_mod import custom_quantity
from cgbench.core import dataset
from cgbench.core.config import DEFAULT_SIM_CONFIG as SIM_CONFIG, BOND_SPRING_CONSTANTS

# -------------------------
# Configuration handling
# -------------------------
model_path = args.model
base_dir = os.path.dirname(model_path)

# Parse manual CG map specification if provided
manual_cg_choice = args.cg_map
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
    manual_cg_choice = "custom"
elif manual_cg_choice is not None and manual_cg_choice.strip().startswith("["):
    custom_cg_str = manual_cg_choice.strip()
    try:
        custom_cg_array = ast.literal_eval(custom_cg_str)
        if not isinstance(custom_cg_array, (list, tuple)):
            raise ValueError("Custom CG map must be a list or tuple of integers.")
        custom_cg_array = [int(x) for x in custom_cg_array]
    except Exception as e:
        raise ValueError(f"Failed to parse --cg-map array string '{manual_cg_choice}': {e}")
    manual_cg_choice = "custom"

# Assign unique map tag to avoid directory collisions between different custom maps
if custom_cg_array is not None:
    valid_indices = [idx for idx in custom_cg_array if idx >= 0]
    n_beads = max(valid_indices) + 1 if valid_indices else 0
    if args.map_name:
        map_tag = args.map_name.strip()
    else:
        import hashlib
        map_hash = hashlib.md5(str(custom_cg_array).encode()).hexdigest()[:6]
        map_tag = f"custom_{n_beads}beads_{map_hash}"
    manual_cg_choice = map_tag
elif args.map_name and manual_cg_choice is not None:
    manual_cg_choice = args.map_name.strip()

# Load MACE config
mace_config_path = os.path.join(base_dir, "config.json")
if os.path.exists(mace_config_path):
    with open(mace_config_path, "r") as f:
        # load
        MACE_CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"Config file {mace_config_path} not found.")

# Load training config
train_config_path = os.path.join(base_dir, "train_config.json")
if os.path.exists(train_config_path):
    with open(train_config_path, "r") as f:
        TRAIN_CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"Train config file {train_config_path} not found.")

if manual_cg_choice is not None and not MACE_CONFIG.get("learned_species_embedding", False):
    raise NotImplementedError(
        f"Simulating with a manual CG map ('{manual_cg_choice}') is only supported for models trained "
        f"with learned species embeddings composed of learned atom embeddings (learned_species_embedding=True)."
    )

# -------------------------
# Load model checkpoint
# -------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")

energy_params_raw = onp.load(model_path, allow_pickle=True)
energy_params_raw = tree_util.tree_map(jnp.asarray, energy_params_raw)

if isinstance(energy_params_raw, dict) and "mace" in energy_params_raw:
    energy_params = energy_params_raw["mace"]
    cg_params = energy_params_raw.get("cg_map", None)
else:
    energy_params = energy_params_raw
    cg_params = None

is_learned = (MACE_CONFIG.get("CG_map") == "learned")
learned_map = None

if is_learned:
    # 1. Prioritize extracting the mapping directly from checkpoint weights to ensure
    # exact consistency with best_params.pkl / final_params.pkl
    if cg_params is not None:
        dense_keys = [k for k in cg_params.keys() if "Dense" in k]
        if dense_keys:
            dense = cg_params[dense_keys[0]]
            k = dense.get("kernel")
            b = dense.get("bias", 0.0)
            logits = (k + b[None, :]) if b is not None else k
            learned_map = [int(x) for x in onp.asarray(jnp.argmax(logits, axis=-1))]
            print(f"[INFO] Extracted learned CG map from checkpoint weights: {learned_map}")

    # 2. Fall back to cg_maps.json if not found in checkpoint weights
    if learned_map is None:
        cg_maps_path = os.path.join(base_dir, "cg_maps.json")
        if os.path.exists(cg_maps_path):
            with open(cg_maps_path, "r") as f:
                cg_maps_dict = json.load(f)
            last_epoch = max(cg_maps_dict.keys(), key=int)
            learned_map = cg_maps_dict[last_epoch]
            print(f"[INFO] Loaded learned CG map from {cg_maps_path} (epoch {last_epoch}): {learned_map}")

    if learned_map is None and manual_cg_choice is None:
        raise ValueError(f"Could not determine learned CG mapping from {base_dir}")

config = SIM_CONFIG.copy()
config["sim_mol"] = args.mol
config["type"] = MACE_CONFIG["type"]
if manual_cg_choice is not None:
    config["cg_map"] = manual_cg_choice
else:
    config["cg_map"] = MACE_CONFIG.get("CG_map", None)

if args.t_total is not None:
    config["t_total"] = args.t_total
if args.n_chains is not None:
    config["n_chains"] = args.n_chains
if args.sim_mode is not None:
    config["sim_mode"] = args.sim_mode
if args.dt_fs is not None:
    config["dt_values_fs"] = args.dt_fs
if args.ensemble is not None:
    config["ensemble"] = args.ensemble

if args.verbose:
    print("-" * 50)
    for key, value in MACE_CONFIG.items():
        print(f"Found MACE config: {key}: {value}")
    print("-" * 50)
    for key, value in config.items():
        print(f"Using Sim config: {key}: {value}")
    print("-" * 50)

# -------------------------
# Load dataset
# -------------------------
if MACE_CONFIG["mol"] == "ala2":
    data = dataset.Ala2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"],
        val_ratio=MACE_CONFIG["val_ratio"],
    )
    MACE_CONFIG["nmol"] = 1
elif MACE_CONFIG["mol"] == "hexane":
    data = dataset.Hexane_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"],
        val_ratio=MACE_CONFIG["val_ratio"],
    )
    MACE_CONFIG["nmol"] = 100
elif MACE_CONFIG["mol"] == "ala15":
    data = dataset.Ala15_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
    MACE_CONFIG["nmol"] = 1
elif MACE_CONFIG["mol"] == "pro2":
    data = dataset.Pro2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
    MACE_CONFIG["nmol"] = 1
elif MACE_CONFIG["mol"] == "thr2":
    data = dataset.Thr2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
    MACE_CONFIG["nmol"] = 1
elif MACE_CONFIG["mol"] == "gly2":
    data = dataset.Gly2_Dataset(
        train_ratio=MACE_CONFIG["train_ratio"], val_ratio=MACE_CONFIG["val_ratio"]
    )
    MACE_CONFIG["nmol"] = 1
else:
    raise ValueError(
        "Invalid molecule. Use 'ala2', 'ala15', 'hexane', 'pro2', 'thr2', or 'gly2'."
    )

# AT
if MACE_CONFIG["type"] == "AT":
    dataset = data.dataset_U
    species = data.species
    masses = data.masses
    n_species = data.n_species
    num_cg_beads = len(species)
    map_indices = None
# CG
elif MACE_CONFIG["type"] == "CG":
    from cgbench.core.mapping import register_custom_map
    if manual_cg_choice is not None:
        if custom_cg_array is not None:
            register_custom_map(data.map_obj, custom_cg_array)
            if manual_cg_choice != "custom":
                data.map_obj._maps[manual_cg_choice] = data.map_obj._maps["custom"]
            data.coarse_grain(map="custom")
            dataset = data.cg_dataset_U
            masses = data.cg_masses
            num_cg_beads = data.n_cg_sites
            map_indices = custom_cg_array
            print(f"[INFO] Coarse-grained using manual custom map '{manual_cg_choice}': {map_indices} ({num_cg_beads} beads)")
        else:
            data.coarse_grain(map=manual_cg_choice)
            dataset = data.cg_dataset_U
            masses = data.cg_masses
            num_cg_beads = data.n_cg_sites
            map_indices, _, _, _ = data.map_obj.get_map(manual_cg_choice)
            print(f"[INFO] Coarse-grained using manual map '{manual_cg_choice}': {map_indices} ({num_cg_beads} beads)")
    elif is_learned:
        register_custom_map(data.map_obj, learned_map)
        data.coarse_grain(map="custom")
        dataset = data.cg_dataset_U
        masses = data.cg_masses
        num_cg_beads = data.n_cg_sites
        map_indices = learned_map
        print(f"[INFO] Coarse-grained using learned map: {map_indices} ({num_cg_beads} beads)")
    else:
        data.coarse_grain(map=MACE_CONFIG["CG_map"])
        dataset = data.cg_dataset_U
        species = data.cg_species
        masses = data.cg_masses
        n_species = data.n_cg_species
        num_cg_beads = len(species)
        map_indices = None
else:
    raise ValueError("Invalid simulation type. Use 'AT' or 'CG'.")

# -------------------------
# Species resolution
# -------------------------
if MACE_CONFIG["type"] == "CG" and (is_learned or manual_cg_choice is not None):
    if MACE_CONFIG.get("learned_species_embedding", False):
        dim = MACE_CONFIG.get("species_embedding_dim", 16)
        n_species = dim

        if cg_params is not None and "atom_type_embeddings" in cg_params:
            atom_type_embeddings = jnp.asarray(cg_params["atom_type_embeddings"])
        else:
            atom_emb_path = os.path.join(base_dir, "atom_embeddings.json")
            if os.path.exists(atom_emb_path):
                with open(atom_emb_path, "r") as f:
                    saved_atom_emb = json.load(f)
                last_ep = max(saved_atom_emb.keys(), key=int)
                emb_dict = saved_atom_emb[last_ep]
                atom_type_embeddings = jnp.asarray([emb_dict[str(k)] for k in sorted([int(x) for x in emb_dict.keys()])])
                print(f"[INFO] Loaded atom type embeddings from {atom_emb_path} (epoch {last_ep})")
            else:
                raise ValueError(f"Could not find atom_type_embeddings in checkpoint or {atom_emb_path}")

        if MACE_CONFIG.get("normalize_atom_embedding", False):
            norm = jnp.linalg.norm(atom_type_embeddings, axis=-1, keepdims=True)
            atom_type_embeddings = atom_type_embeddings / (norm + 1e-8)

        # Count atom types per bead
        atom_species_all = data.dataset_U["training"]["species"][0]
        unique_atom_species = jnp.array(sorted(list(set(atom_species_all.tolist()))))
        num_atom_types = unique_atom_species.shape[0]
        atom_type_indices = jnp.argmax(atom_species_all[:, None] == unique_atom_species[None, :], axis=-1)
        one_hot_atom_types = jax.nn.one_hot(atom_type_indices, num_atom_types)

        map_arr = jnp.asarray(map_indices, dtype=jnp.int32)
        valid = map_arr >= 0
        clipped = jnp.where(valid, map_arr, 0)
        assign_one_hot = jax.nn.one_hot(clipped, num_cg_beads, dtype=jnp.float32) * valid[:, None]
        assign_t = assign_one_hot.T
        bead_atom_types = jnp.matmul(assign_t, one_hot_atom_types)

        species_init = jnp.matmul(bead_atom_types, atom_type_embeddings)
        print(f"[INFO] Initialized species features from learned atom embeddings: shape {species_init.shape}, dim {dim}")

    elif manual_cg_choice is not None:
        raise NotImplementedError(
            f"Simulating with a manual CG map ('{manual_cg_choice}') is only supported for models trained "
            f"with learned species embeddings composed of learned atom embeddings (learned_species_embedding=True)."
        )

    elif MACE_CONFIG.get("unique_cg_species", False):
        n_species = num_cg_beads
        species_init = jnp.arange(1, num_cg_beads + 1, dtype=jnp.int32)
        print(f"[INFO] Using unique CG species IDs: 1..{num_cg_beads}")

    else:
        cg_species_saved = MACE_CONFIG.get("cg_species", data.cg_species)
        _, species_dense = onp.unique(cg_species_saved, return_inverse=True)
        species_init = jnp.asarray(species_dense + 1, dtype=jnp.int32)
        n_species = len(set(species_init.tolist()))
        print(f"[INFO] Using categorical CG species: {n_species} unique classes")

elif MACE_CONFIG["type"] == "CG":
    _, species_dense = onp.unique(species, return_inverse=True)
    species_init = jnp.asarray(species_dense + 1, dtype=jnp.int32)
    n_species = len(set(species_init.tolist()))
else:
    _, species_dense = onp.unique(species, return_inverse=True)
    species_init = jnp.asarray(species_dense + 1, dtype=jnp.int32)
    n_species = len(set(species_init.tolist()))

# -------------------------
# Neighbor list setup
# -------------------------
r_cutoff = MACE_CONFIG["r_cutoff"]
box = data.box

displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
    preprocessing.allocate_neighborlist(
        dataset["training"],
        displacement_fn,
        box,
        r_cutoff=MACE_CONFIG["r_cutoff"],
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=100,
        capacity_multiplier=1.0,
    )
)

# -------------------------
# Model initialization
# -------------------------
mace_cfg = {
    "atomic_numbers": onp.arange(1, n_species + 1, dtype=onp.int32),
    "r_cutoff": MACE_CONFIG["r_cutoff"],
    "hidden_irreps": MACE_CONFIG["hidden_irreps"],
    "MLP_irreps": MACE_CONFIG["readout_mlp_irreps"],
    "num_interactions": MACE_CONFIG["num_interactions"],
    "max_ell": MACE_CONFIG["max_ell"],
    "correlation": MACE_CONFIG["correlation"],
    "n_radial_basis": MACE_CONFIG["n_radial_basis"],
    "output_irreps": MACE_CONFIG["output_irreps"],
    "use_so3": MACE_CONFIG.get("use_so3", False),
}

cueq_config = CuEquivarianceConfig(
    enabled=True,
    layout=("mul_ir"),
    group=("O3"),
    optimize_all=True,
    conv_fusion=True,
)
if MACE_CONFIG.get("use_so3", False):
    print("[NOTE] Using SO(3) equivariance (no CuEquivariance support)")
    cueq_config = None

template_vars, gnn_energy_fn, model_config = mace_jax_compose.mace_jax_neighborlist(
    displacement=displacement_fn,
    r_cutoff=MACE_CONFIG["r_cutoff"],
    n_species=n_species,
    per_particle=False,
    avg_num_neighbors=avg_num_neighbors,
    mode="energy",
    use_custom_batch_fn=False,
    mace_config=mace_cfg,
    cueq_config=cueq_config,
)

variables = template_vars

if 'use_bond_priors' in MACE_CONFIG and MACE_CONFIG['use_bond_priors']:
    print("Using bond priors in simulation energy function.")

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
    
    if MACE_CONFIG.get("use_bond_priors", False):
        sim_map_name = config["cg_map"]
        key = f"mol={MACE_CONFIG['mol']}_map={sim_map_name}"
        if key not in BOND_SPRING_CONSTANTS:
            fallback_key = f"mol={MACE_CONFIG['mol']}_map=hmerged"
            if fallback_key in BOND_SPRING_CONSTANTS:
                key = fallback_key
                print(f"[WARNING] Prior constants for '{sim_map_name}' not found. Falling back to '{fallback_key}'.")
        if key in BOND_SPRING_CONSTANTS:
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
            print(f"[WARNING] No prior constants found for key '{key}'. Proceeding without bond priors.")
            return energy_fn
    else:
        return energy_fn

if args.verbose:
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")

energy_fn = energy_fn_template(energy_params)

# Setup base dir
if manual_cg_choice is not None:
    outdir = f"{base_dir}/simulation_{config['ensemble']}_T={config['T']}K_map={config['cg_map']}/"
else:
    outdir = f"{base_dir}/simulation_{config['ensemble']}_T={config['T']}K/"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# Simulator initialization
# -------------------------
def init_simulator(
    dataset, energy_fn, masses, nbrs_init, kT, dt, n_chains, gamma, t_eq, t_total
):
    key = random.PRNGKey(config["PRNGKey_seed"])
    _, shift_fn = space.periodic_general(
        dataset["validation"]["box"][0], fractional_coordinates=True
    )

    if config["sim_mode"] == "stability": # take the first structure and repeat for n_chains
        r_init = dataset["validation"]["R"][0]
        # repeat for n_chains
        r_init = jnp.tile(r_init, (n_chains, 1, 1))

    elif config["sim_mode"] == "helix": # use predefined helix indices
        indices = onp.load(
            "/home/franz/Ala15_100_min_helix_indices.npy", allow_pickle=True
        )
        combined_dataset = onp.concatenate(
            [dataset["training"]["R"], dataset["validation"]["R"]], axis=0
        )

        # assert that the indices are within the range of the combined dataset
        assert (
            jnp.max(indices) < combined_dataset.shape[0]
        ), "Indices exceed dataset size."

        r_init = combined_dataset[indices]
        
    elif config["sim_mode"] == "speed": # random selection, single chain
        key, split = random.split(key)
        selection = random.choice(
            split,
            jnp.arange(dataset["validation"]["R"].shape[0]),
            shape=(1,),
            replace=False,
        )
        r_init = dataset["validation"]["R"][selection]

    else: # sampling mode: random selection, n_chains
        key, split = random.split(key)
        selection = random.choice(
            split,
            jnp.arange(dataset["validation"]["R"].shape[0]),
            shape=(n_chains,),
            replace=False,
        )
        r_init = dataset["validation"]["R"][selection]

    if config["ensemble"] == "NVT":
        init_simulator_fn = simulate.nvt_langevin
        sim_kwargs = {"kT": kT, "gamma": gamma, "dt": dt}
        init_sim_kwargs = {"mass": masses, "neighbor": nbrs_init}

    elif config["ensemble"] == "NVE":
        init_simulator_fn = simulate.nve
        sim_kwargs = {"dt": dt, "kT": kT}
        init_sim_kwargs = {"mass": masses, "neighbor": nbrs_init, "kT": kT}

    else:
        raise ValueError(f"Unknown ensemble: {config['ensemble']}. Use NVT or NVE.")

    init_ref_state, sim_template = sampling.initialize_simulator_template(
        init_simulator_fn,
        shift_fn=shift_fn,
        nbrs=nbrs_init,
        init_with_PRNGKey=True,
        extra_simulator_kwargs=sim_kwargs,
    )

    # Init reference state
    key, split = random.split(key)
    reference_state = init_ref_state(
        split, r_init, energy_or_force_fn=energy_fn, init_sim_kwargs=init_sim_kwargs
    )

    # Setup evaluation timings
    eval_timings = sampling.process_printouts(
        time_step=dt,
        total_time=t_total,
        t_equilib=t_eq,
        print_every=config["print_every"],
    )
    
    # Setup quantities to record
    quantities = {
        # "kT": custom_quantity.temperature,
        "epot": custom_quantity.energy_wrapper(lambda _: energy_fn),
        # "force": custom_quantity.force_wrapper(lambda _: energy_fn),
        # "etot": custom_quantity.total_energy_wrapper(lambda _: energy_fn),
        # "eprior": prior_energy_wrapper(lambda _: harmonic_energy_fn),
    }

    # Initialize trajectory generator
    traj_gen = sampling.trajectory_generator_init(
        sim_template,
        lambda _: energy_fn,
        eval_timings,
        quantities=quantities,
        vmap_sim_batch=config["n_chains"],
        vmap_batch=config["n_chains"],
    )

    return reference_state, jax.jit(traj_gen)


def visualise(traj_path, dataset):
    from cgbench.plotting import molecules as visualise_traj
    vis_fn_map = {
        'ala2': visualise_traj.vis_ala2,
        'hexane': visualise_traj.vis_hexane,
        'ala15': visualise_traj.vis_ala15,
        'pro2': visualise_traj.vis_pro2,
        'thr2': visualise_traj.vis_thr2,
        'gly2': visualise_traj.vis_gly2,
    }

    if config['sim_mol'] in vis_fn_map:
        vis_fn = vis_fn_map[config['sim_mol']]
        vis_fn(traj_path, config, type=MACE_CONFIG['type'], dataset=dataset, cg_map=config['cg_map'])
    else:
        raise ValueError("Invalid molecule. Use 'ala2', 'hexane', 'ala15', 'pro2', 'thr2', or 'gly2'.")


dt_values_fs = config["dt_values_fs"]
dt_values_ps = [dt_fs * 0.001 for dt_fs in dt_values_fs]  # convert to ps

for dt_fs, dt_ps in zip(dt_values_fs, dt_values_ps):
    # Update config for this dt
    config["dt"] = dt_ps

    print(f"\nStarting simulation for dt = {dt_fs} fs ({dt_ps} ps)...")
    folder_name = f"traj_mol={config['sim_mol']}_dt={dt_fs}_teq={config['t_eq']}_t={config['t_total']}_nmol={MACE_CONFIG['nmol']}_nchains={config['n_chains']}_mode={config['sim_mode']}_seed={config['PRNGKey_seed']}/"
    save_dir = os.path.join(outdir, folder_name)

    # Skip simulation if folder already exists
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(
            f"Directory {save_dir} already exists and is not empty. Skipping simulation for dt = {dt_fs} fs."
        )
        visualise(save_dir, data)
        continue
    os.makedirs(save_dir, exist_ok=True)

    reference_state, traj_generator = init_simulator(
        dataset,
        energy_fn,
        masses,
        nbrs_init,
        kT=config["kT"],
        dt=dt_ps,
        n_chains=config["n_chains"],
        gamma=config["gamma"],
        t_eq=config["t_eq"],
        t_total=config["t_total"],
    )

    # time the trajectory generation
    start_time = time.time()
    traj_state = traj_generator(None, reference_state)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")
    
    # calculate ns/day
    total_sim_time_ps = config["t_total"] * config["n_chains"]
    ns_per_day = (total_sim_time_ps / elapsed_time) * (86400 / 1000)
    print(f"Performance: {ns_per_day:.2f} ns/day")

    # Save trajectory
    with open(os.path.join(save_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(traj_state.trajectory.position, f)

    with open(os.path.join(save_dir, "traj_state_aux.pkl"), "wb") as f:
        pickle.dump(traj_state.aux, f)

    config_ = config.copy()
    config_["dt"] = dt_ps
    config_.pop("dt_values_fs", None)
    if custom_cg_array is not None:
        config_["custom_cg_indices"] = custom_cg_array

    # Save config used
    with open(os.path.join(save_dir, "traj_config.json"), "w") as cf:
        json.dump(config_, cf, indent=4)

    print(f"Finished dt = {dt_fs} fs. Results saved to {save_dir}.")

    try:
        visualise(save_dir, data)
    except Exception as e:
        print(f"Error during visualisation: {e}")
