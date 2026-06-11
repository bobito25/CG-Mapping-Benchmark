import argparse
import os
import sys

# Add parent directory to path to import cgbench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--cgmap", type=str, help="CG mapping to use", required=True)
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
args = parser.parse_args()

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"

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

MACE_CONFIG["r_cutoff"] = args.rcut
MACE_CONFIG["mol"] = args.mol 
MACE_CONFIG["CG_map"] = args.cgmap
MACE_CONFIG["use_bond_priors"] = args.prior
MACE_CONFIG["type"] = "CG" if MACE_CONFIG["CG_map"] != "AT" else "AT"

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
    
# We always load the atomistic dataset for learned coarse-graining
# since we need the atomistic positions R and forces F to map from.
dataset = data.dataset_U
species = data.species
masses = data.masses
n_species = data.n_species

output_dir = f"outputs/MLP_train/{MACE_CONFIG['mol'].capitalize()}_map={MACE_CONFIG['CG_map']}_tr={MACE_CONFIG['train_ratio']}_rcut={MACE_CONFIG['r_cutoff']}_epochs={TRAIN_CONFIG['num_epochs']}_int={MACE_CONFIG['num_interactions']}_corr={MACE_CONFIG['correlation']}_seed={MACE_CONFIG['PRNGKey_seed']}_prior={MACE_CONFIG['use_bond_priors']}"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Setup neighbor list and MACE model
# -------------------------
box = data.box
displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

# Lookup target map and num_cg_beads
num_cg_beads = 10
initial_mapping = None
target_map_name = args.cgmap
if target_map_name == "AT":
    target_map_name = "hmerged"  # Default heuristic map for learned CG mapping

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
            map_inst = map_class()
            # If target_map_name is not in the maps, look for a 10-bead map
            if target_map_name not in map_inst.get_available_maps():
                for map_name in map_inst.get_available_maps():
                    indices, cg_species, _, _ = map_inst.get_map(map_name)
                    if len(cg_species) == 10:
                        target_map_name = map_name
                        break
            
            if target_map_name in map_inst.get_available_maps():
                indices, cg_species, _, _ = map_inst.get_map(target_map_name)
                num_cg_beads = len(cg_species)
                indices_clean = [idx if idx >= 0 else 0 for idx in indices]
                initial_mapping = tuple(indices_clean)
                print(f"[INFO] Initializing learned CG mapping using heuristic '{target_map_name}' map: {initial_mapping}")
                print(f"[INFO] Number of CG beads: {num_cg_beads}")
            else:
                print(f"[WARNING] No map found. Using default 10 beads.")
        except Exception as e:
            print(f"[WARNING] Error reading heuristic map for initialization: {e}")

# Create initial mapping matrix for neighbor list allocation
initial_map_arr = jnp.array(initial_mapping, dtype=jnp.int32)
one_hot_map = jax.nn.one_hot(initial_map_arr, num_cg_beads).T  # [num_cg_beads, num_atoms]
atoms_per_bead = jnp.sum(one_hot_map, axis=-1, keepdims=True) + 1e-8
c_map_init = one_hot_map / atoms_per_bead

displacement_fn_X, shift_fn_X = space.periodic_general(box=box, fractional_coordinates=True)
cg_positions_init, _ = mapping.map_dataset(
    dataset["training"]["R"],
    displacement_fn,
    shift_fn_X,
    c_map_init,
    d_map=c_map_init,
    force_dataset=jnp.zeros_like(dataset["training"]["R"])
)

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

if args.verbose:
    print(
        f"Max neighbors: {max_neighbors}, Max edges: {max_edges}, Avg neighbors: {avg_num_neighbors}"
    )

mace_cfg = {
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
cg_species_init = jnp.zeros(num_cg_beads, dtype=jnp.int32)

def energy_fn_template(energy_params):
    vars = {**variables}
    vars["params"] = energy_params

    def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
        del mode
        dynamic_kwargs.setdefault("species", cg_species_init)
        dynamic_kwargs.setdefault("box", box)
        mask = dynamic_kwargs.pop("mask", jnp.ones(pos.shape[0], dtype=jnp.bool_))

        pots = gnn_energy_fn(vars, pos, neighbor, **dynamic_kwargs)
        if pots.ndim == 2 and pots.shape[-1] == 1:
            pots = pots.squeeze(-1)

        atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
        atomic_energies = jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)
        mapped_species = jnp.argmax(dynamic_kwargs["species"][:, None] == atomic_numbers[None, :], axis=-1)

        pots = (pots - atomic_energies[mapped_species]) * mask
        return jnp.sum(pots)
    
    if args.prior:
        key = f"mol={MACE_CONFIG['mol']}_map={MACE_CONFIG['CG_map']}"
        assert key in BOND_SPRING_CONSTANTS
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
r_init_cg = cg_positions_init[0]
mask_init_cg = jnp.ones(num_cg_beads, dtype=jnp.bool_)
nbrs_init = nbrs_init.update(r_init_cg, mask=mask_init_cg)

cg_model = GumbelCGAssignment(num_cg_beads=num_cg_beads, initial_mapping=initial_mapping)
cg_params = cg_model.init(
    jax.random.PRNGKey(0),
    r_init,
    dataset["training"]["species"][0],
    jax.random.PRNGKey(1)
)

#def combined_energy_fn_template(joint_params):
#    # Pass in both CG mapping and MACE params inside a tree
#    mace_energy_fn = energy_fn_template(joint_params['mace'])
#    
#    # We must construct a wrapper that takes the ATOMISTIC positions 
#    # but delegates to MACE after mapping
#    def forward_call(pos, neighbor, mode=None, **dynamic_kwargs):
#        # 1. Pull the stochastic key from kwargs (injected by updated trainer step)
#        prng_key = dynamic_kwargs.get("prng_key", jax.random.PRNGKey(0))
#        
#        # 2. Get Softmax Map
#        c_map = cg_model.apply({'params': joint_params['cg_map']}, pos, dynamic_kwargs['node_attrs'], prng_key)
#        
#        # 3. Map Coordinates Dynamically (in-tape!)
#        displacement_fn, shift_fn_X = space.periodic_general(box=box, fractional_coordinates=False)
#        pos_cg, _ = mapping.map_dataset(pos, displacement_fn, shift_fn_X, c_map, d_map=c_map)
#        
#        # 4. Detach for Neighbor updates
#        cg_nbrs = neighbor.update(jax.lax.stop_gradient(pos_cg))
#        
#        # 5. Delegate to MACE
#        return mace_energy_fn(pos_cg, cg_nbrs, mode=mode, **dynamic_kwargs)
#        
#    return forward_call

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
joint_params = {'mace': init_params, 'cg_map': cg_params['params']}
trainer_fm = CGForceMatching(
    joint_params,
    optimizer_fm,
    energy_fn_template,
    nbrs_init,
    log_file=f"{output_dir}/force_matching.log",
    batch_per_device=int(batch_size),
    model_cg_map=cg_model,
)

def print_learned_mapping(trainer, *args, **kwargs):
    cg_map_params = trainer.params.get('cg_map', None)
    if cg_map_params is not None:
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
                epoch = trainer._epoch
                print(f"\n--- Learned CG Mapping at Epoch {epoch} ---")
                
                # Print kernel stats
                k_min, k_max, k_mean = jnp.min(kernel), jnp.max(kernel), jnp.mean(kernel)
                print(f"[DEBUG] Mapping weights stats - Min: {k_min:.4f}, Max: {k_max:.4f}, Mean: {k_mean:.4f}")
                
                assignments_list = [int(x) for x in jax.device_get(assignments)]
                print(f"Atom assignments to CG beads: {assignments_list}")
                
                bead_to_atoms = {}
                for atom_idx, bead_idx in enumerate(assignments_list):
                    bead_to_atoms.setdefault(bead_idx, []).append(atom_idx)
                
                print("CG Beads to Atoms:")
                for bead_idx in sorted(bead_to_atoms.keys()):
                    print(f"  Bead {bead_idx}: Atoms {bead_to_atoms[bead_idx]}")
                print("-------------------------------------------\n", flush=True)

trainer_fm.add_task("post_epoch", print_learned_mapping)
trainer_fm.set_dataset(dataset["training"], stage="training")
trainer_fm.set_dataset(dataset["validation"], stage="validation", include_all=True)
if "testing" in dataset:
    trainer_fm.set_dataset(dataset["testing"], stage="testing", include_all=True)

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

from cgbench.plotting.training import plot_predictions, plot_convergence

# Plot training convergence
plot_convergence(trainer_fm, output_dir)

predictions_val = trainer_fm.predict(
    dataset["validation"],
    trainer_fm.best_params,
    batch_size=batch_size,
)
predictions_val = tree_util.tree_map(onp.asarray, predictions_val)
plot_predictions(
    predictions_val, dataset["validation"], output_dir, name="preds_validation"
)

if "testing" in dataset:
    predictions_test = trainer_fm.predict(
        dataset["testing"],
        trainer_fm.best_params,
        batch_size=batch_size,
    )
    predictions_test = tree_util.tree_map(onp.asarray, predictions_test)
    onp.savez(f"{output_dir}/predictions_test.npz", **predictions_test)
    plot_predictions(
        predictions_test, dataset["testing"], output_dir, name="preds_testing"
    )
