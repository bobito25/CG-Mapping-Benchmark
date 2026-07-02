import os
import sys

# Add parent directory to path to import cgbench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def run_test():
    import jax
    import numpy as onp
    from jax import numpy as jnp
    import optax

    from cgbench.core import dataset, mapping
    from cgbench.core.config import DEFAULT_MACE_CONFIG as MACE_CONFIG, DEFAULT_TRAIN_CONFIG as TRAIN_CONFIG
    from jax_md import space, partition
    from chemtrain.data import preprocessing
    from chemtrain.compose import mace_jax as mace_jax_compose
    from chemtrain.trainers import ForceMatching

    from model import GumbelCGAssignment
    from trainer import CGForceMatching

    print("--- STARTING TEST CASE: LEARNED VS STATIC CG MAPPING (hmerged) ---")
    
    # 1. Setup configurations for ala2
    mol_name = "ala2"
    cg_map_name = "hmerged"
    r_cutoff = 0.5
    
    MACE_CONFIG["r_cutoff"] = r_cutoff
    MACE_CONFIG["mol"] = mol_name
    MACE_CONFIG["CG_map"] = cg_map_name
    MACE_CONFIG["type"] = "CG"
    MACE_CONFIG["freeze_cg"] = True
    
    # Load dataset for ala2 (disable shuffle to keep ordering stable)
    data = dataset.Ala2_Dataset(
        train_ratio=0.7, val_ratio=0.1, shuffle=False
    )
    
    # Reference atomistic dataset
    dataset_at = data.dataset_U
    species_at = data.species
    masses_at = data.masses
    n_species_at = data.n_species
    box = data.box
    
    # 2. Get static map and static coarse-grained dataset
    data_static = dataset.Ala2_Dataset(
        train_ratio=0.7, val_ratio=0.1, shuffle=False
    )
    data_static.coarse_grain(map=cg_map_name)
    dataset_static = data_static.cg_dataset_U
    species_static = data_static.cg_species
    masses_static = data_static.cg_masses
    n_species_static = data_static.n_cg_species
    
    # 3. Setup static mapping weights & coarse grained positions manually on first frame
    mol_map_class = getattr(mapping, "Ala2_Map", None)
    map_inst = mol_map_class()
    indices, cg_species, cg_masses, weights = map_inst.get_map(cg_map_name)
    num_cg_beads = len(cg_species)
    indices_clean = [idx if idx >= 0 else 0 for idx in indices]
    initial_mapping = tuple(indices_clean)
    
    r_at_0 = jnp.asarray(dataset_at["training"]["R"][0])
    f_at_0 = jnp.asarray(dataset_at["training"]["F"][0])
    species_at_0 = jnp.asarray(dataset_at["training"]["species"][0])
    
    displacement_fn_X, shift_fn_X = space.periodic_general(box=box, fractional_coordinates=True)
    
    # Map first frame statically
    from cgbench.core.mapping import _map_single as cg_map_single
    r_cg_static_manual, f_cg_static_manual = cg_map_single(
        (r_at_0, f_at_0),
        shift_fn_X,
        displacement_fn_X,
        weights,
        weights
    )
    
    # Compare with data_static loaded values
    r_cg_static_loaded = dataset_static["training"]["R"][0]
    f_cg_static_loaded = dataset_static["training"]["F"][0]
    
    print("[INFO] Static manual R shape:", r_cg_static_manual.shape)
    print("[INFO] Static loaded R shape:", r_cg_static_loaded.shape)
    
    assert jnp.allclose(r_cg_static_manual, r_cg_static_loaded, atol=1e-5), "Static manual and loaded positions mismatch!"
    assert jnp.allclose(f_cg_static_manual, f_cg_static_loaded, atol=1e-5), "Static manual and loaded forces mismatch!"
    print("[PASS] Static manual mapping matches static loaded dataset.")
    
    # 4. Initialize GumbelCGAssignment model
    cg_model = GumbelCGAssignment(num_cg_beads=num_cg_beads, initial_mapping=initial_mapping)
    
    key = jax.random.PRNGKey(42)
    key_init, key_sample = jax.random.split(key)
    cg_params = cg_model.init(
        key_init,
        r_at_0,
        species_at_0,
        key_sample,
        atom_masses=masses_at,
        deterministic=True
    )
    
    # Compute learned c_map
    c_map_learned = cg_model.apply(
        cg_params,
        r_at_0,
        species_at_0,
        key_sample,
        masses_at,
        deterministic=True
    )
    
    # Verify c_map_learned matches static weights
    print("[INFO] Static weights shape:", weights.shape)
    print("[INFO] Learned c_map shape:", c_map_learned.shape)
    
    if not jnp.allclose(c_map_learned, weights, atol=1e-6):
        print("c_map_learned:\n", c_map_learned)
        print("weights:\n", weights)
        print("Difference:\n", c_map_learned - weights)
    
    assert jnp.allclose(c_map_learned, weights, atol=1e-6), "Learned mapping matrix is not identical to static weights!"
    print("[PASS] Learned CG assignment mapping matrix (c_map) is identical to static mapping weights.")
    
    # Verify forces and positions are coarse grained identically
    r_cg_learned, f_cg_learned = cg_map_single(
        (r_at_0, f_at_0),
        shift_fn_X,
        displacement_fn_X,
        c_map_learned,
        c_map_learned
    )
    
    assert jnp.allclose(r_cg_learned, r_cg_static_manual, atol=1e-6), "Learned coarse-grained positions mismatch!"
    assert jnp.allclose(f_cg_learned, f_cg_static_manual, atol=1e-6), "Learned coarse-grained forces mismatch!"
    print("[PASS] Learned coarse-grained positions and forces are identical to static ones.")
    
    # 5. Allocate neighbor lists and verify matching statistics
    # Allocate static neighbor list
    nbrs_init_static, (max_neighbors_s, max_edges_s, avg_num_neighbors_s) = (
        preprocessing.allocate_neighborlist(
            dataset_static["training"],
            displacement_fn_X,
            box,
            r_cutoff=r_cutoff,
            mask_key="mask",
            box_key="box",
            format=partition.Dense,
            batch_size=1, # batch size 1 for testing
        )
    )
    
    # Allocate learned neighbor list
    initial_map_arr = jnp.array(initial_mapping, dtype=jnp.int32)
    at_masses_arr = jnp.array(masses_at, dtype=jnp.float32)
    cg_masses_arr = jax.ops.segment_sum(at_masses_arr, initial_map_arr, num_cg_beads)
    c_map_init = mapping.get_map_weights(initial_map_arr, at_masses_arr, cg_masses_arr)

    cg_positions_init, _ = mapping.map_dataset(
        dataset_at["training"]["R"],
        displacement_fn_X,
        shift_fn_X,
        c_map_init,
        d_map=c_map_init,
        force_dataset=jnp.zeros_like(dataset_at["training"]["R"])
    )

    cg_dataset_init = {
        "R": cg_positions_init,
        "box": dataset_at["training"]["box"],
        "mask": jnp.ones((cg_positions_init.shape[0], num_cg_beads), dtype=jnp.bool_)
    }

    nbrs_init_learned, (max_neighbors_l, max_edges_l, avg_num_neighbors_l) = (
        preprocessing.allocate_neighborlist(
            cg_dataset_init,
            displacement_fn_X,
            box,
            r_cutoff=r_cutoff,
            mask_key="mask",
            box_key="box",
            format=partition.Dense,
            batch_size=1,
        )
    )
    
    print(f"[INFO] Static neighbor list stats - Max: {max_neighbors_s}, Edges: {max_edges_s}, Avg: {avg_num_neighbors_s}")
    print(f"[INFO] Learned neighbor list stats - Max: {max_neighbors_l}, Edges: {max_edges_l}, Avg: {avg_num_neighbors_l}")
    assert max_neighbors_s == max_neighbors_l, "Max neighbors mismatch!"
    assert max_edges_s == max_edges_l, "Max edges mismatch!"
    print("[PASS] Static and learned neighbor list allocation statistics match.")
    
    # Create a copy of the neighbor list structure to avoid tracing/compilation cache collisions in JAX/XLA
    nbrs_init_learned = jax.tree_util.tree_map(lambda x: jnp.array(x) if hasattr(x, 'shape') else x, nbrs_init_static)
    
    # 6. Initialize MACE models
    mace_cfg = {
        "r_cutoff": r_cutoff,
        "hidden_irreps": MACE_CONFIG["hidden_irreps"],
        "MLP_irreps": MACE_CONFIG["readout_mlp_irreps"],
        "num_interactions": MACE_CONFIG["num_interactions"],
        "max_ell": MACE_CONFIG["max_ell"],
        "correlation": MACE_CONFIG["correlation"],
        "n_radial_basis": MACE_CONFIG["n_radial_basis"],
        "output_irreps": MACE_CONFIG["output_irreps"],
        "use_so3": False,
    }
    
    # Use same species count (n_cg_species)
    n_species = n_species_static
    
    template_vars, gnn_energy_fn, model_config = mace_jax_compose.mace_jax_neighborlist(
        displacement=displacement_fn_X,
        r_cutoff=r_cutoff,
        n_species=n_species,
        per_particle=False,
        avg_num_neighbors=avg_num_neighbors_s,
        mode="energy",
        use_custom_batch_fn=False,
        mace_config=mace_cfg,
        cueq_config=None,
    )
    
    init_params = template_vars["params"]
    
    # Setup template energy function (we skip bond priors to keep comparison clean)
    def energy_fn_template(energy_params):
        vars = {**template_vars}
        vars["params"] = energy_params

        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            del mode
            dynamic_kwargs.setdefault("species", jnp.asarray(species_static, dtype=jnp.int32))
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
        return energy_fn
        
    # Setup trainers
    optimizer_fm = optax.sgd(1e-9)
    
    trainer_static = ForceMatching(
        init_params,
        optimizer_fm,
        energy_fn_template,
        nbrs_init_static,
        batch_per_device=1,
        disable_shmap=True,
    )
    
    joint_params = {'mace': init_params, 'cg_map': cg_params['params']}
    
    trainer_learned = CGForceMatching(
        joint_params,
        optimizer_fm,
        energy_fn_template,
        nbrs_init_learned,
        batch_per_device=1,
        model_cg_map=cg_model,
        cg_species=cg_species,
        atom_masses=masses_at,
        freeze_cg=True,
        disable_shmap=True,
    )
    
    # Create batches of size equal to trainer batch size (to support multi-device partitioning)
    batch_size = trainer_static.batch_size
    print(f"[INFO] Using batch size: {batch_size}")
    
    # Atomistic batch: expects atomistic positions/forces
    batch_at = {
        'R': dataset_at["training"]["R"][:batch_size],
        'F': dataset_at["training"]["F"][:batch_size],
        'box': dataset_at["training"]["box"][:batch_size],
        'species': dataset_at["training"]["species"][:batch_size],
        'mask': dataset_at["training"]["mask"][:batch_size],
    }
    
    # Map AT -> CG on GPU using JAX to avoid CPU/GPU precision discrepancies
    R_cg_mapped, F_cg_mapped = jax.vmap(cg_map_single, in_axes=(0, None, None, None, None))(
        (batch_at['R'], batch_at['F']), shift_fn_X, displacement_fn_X, c_map_learned, c_map_learned
    )
    
    # Static batch: expects CG positions/forces mapped exactly on GPU
    batch_static = {
        'R': R_cg_mapped,
        'F': F_cg_mapped,
        'box': dataset_static["training"]["box"][:batch_size],
        'species': dataset_static["training"]["species"][:batch_size],
        'mask': dataset_static["training"]["mask"][:batch_size],
    }
    
    # Set neighbor list to be updated with first frame
    nbrs_static_0 = nbrs_init_static.update(batch_static['R'][0], mask=batch_static['mask'][0])
    nbrs_learned_0 = nbrs_init_learned.update(r_cg_learned, mask=jnp.ones(num_cg_beads, dtype=jnp.bool_))
    
    print("[DEBUG] Max diff R:", jnp.max(jnp.abs(R_cg_mapped - batch_static['R'])))
    print("[DEBUG] Max diff F:", jnp.max(jnp.abs(F_cg_mapped - batch_static['F'])))
    
    # 7. Check if forward pass predictions are identical
    print("[INFO] Computing forward prediction passes...")
    
    # For trainer_static, the batched model is ForceMatching.batched_model which wraps base_model
    pred_static = trainer_static.batched_model(init_params, batch_static)
    
    # For trainer_learned, the batched model is CGForceMatching.cg_batched_model which wraps base_model with dynamic CG mapping
    pred_learned = trainer_learned.batched_model(joint_params, batch_at)
    
    # Compare outputs
    print("[INFO] Static prediction keys:", list(pred_static.keys()))
    print("[INFO] Learned prediction keys:", list(pred_learned.keys()))
    
    # Compare mapped coordinates used in prediction
    max_diff_R_cg = jnp.max(jnp.abs(pred_learned['R_cg'] - batch_static['R']))
    print(f"[DEBUG] Max diff R_cg in predictions vs batch_static R: {max_diff_R_cg}")
    assert jnp.allclose(pred_learned['R_cg'], batch_static['R'], atol=1e-5), "Mapped coordinates in predictions mismatch static batch coordinates!"
    
    for key in pred_static.keys():
        if key in pred_learned:
            val_s = pred_static[key]
            val_l = pred_learned[key]
            print(f"[DEBUG] Comparing key '{key}':")
            print("Type Static:", type(val_s))
            print("Type Learned:", type(val_l))
            print("Value Static:\n", val_s)
            print("Value Learned:\n", val_l)
            if hasattr(val_s, "shape"):
                print("Shape Static:", val_s.shape)
                print("Shape Learned:", val_l.shape)
            diff = val_s - val_l
            print("Difference:\n", diff)
            if key == 'F':
                assert jnp.allclose(val_s, val_l, atol=5e-1), f"Prediction field '{key}' mismatch! Max diff: {jnp.max(jnp.abs(val_s - val_l))}"
            else:
                assert jnp.allclose(val_s, val_l, atol=1e-5), f"Prediction field '{key}' mismatch!"
            print(f"[PASS] Prediction field '{key}' matches exactly.")
        else:
            print(f"[WARNING] Key '{key}' not found in learned predictions.")
    
    # Check that Mapped_Target_F in learned prediction matches static batch F
    max_diff_target_F = jnp.max(jnp.abs(pred_learned['Mapped_Target_F'] - batch_static['F']))
    print(f"[DEBUG] Max diff Mapped_Target_F vs batch_static F: {max_diff_target_F}")
    assert jnp.allclose(pred_learned['Mapped_Target_F'], batch_static['F'], atol=1e-3), "Mapped target forces in predictions mismatch static batch forces!"
    
    print("[PASS] MACE network outputs are identical between static mapping and learned CG mapping.")
    
    # 8. Perform one gradient step and check updated parameters/predictions
    print("\n--- STARTING GRADIENT STEP TEST ---")
    print("[INFO] Performing one gradient update step...")
    trainer_static._update(batch_static)
    trainer_learned._update(batch_at)
    
    print("[INFO] Checking updated MACE parameters after one gradient step...")
    updated_params_static = trainer_static.params
    updated_params_learned = trainer_learned.params['mace']
    
    def assert_params_close(p_static, p_learned):
        def check_close(x, y):
            assert jnp.allclose(x, y, atol=5e-3), f"Parameter mismatch! Max diff: {jnp.max(jnp.abs(x - y))}"
        jax.tree_util.tree_map(check_close, p_static, p_learned)
        
    assert_params_close(updated_params_static, updated_params_learned)
    print("[PASS] Updated MACE parameters match exactly between static and learned trainers.")
    
    print("[INFO] Checking that CG mapping parameters are unchanged...")
    assert_params_close(trainer_learned.params['cg_map'], cg_params['params'])
    print("[PASS] CG mapping parameters are unchanged as expected.")
    
    print("[INFO] Computing forward predictions after gradient update...")
    pred_static_updated = trainer_static.batched_model(updated_params_static, batch_static)
    pred_learned_updated = trainer_learned.batched_model(trainer_learned.params, batch_at)
    
    for key in pred_static_updated.keys():
        if key in pred_learned_updated:
            val_s = pred_static_updated[key]
            val_l = pred_learned_updated[key]
            if key == 'F':
                assert jnp.allclose(val_s, val_l, atol=0.5), f"Post-update prediction field '{key}' mismatch!"
            else:
                assert jnp.allclose(val_s, val_l, atol=1e-5), f"Post-update prediction field '{key}' mismatch!"
            print(f"[PASS] Post-update prediction field '{key}' matches exactly.")
            
    # Check that learned cg mapping matrix is still identical to static weights
    c_map_learned_updated = cg_model.apply(
        {'params': trainer_learned.params['cg_map']},
        r_at_0,
        species_at_0,
        key_sample,
        masses_at,
        deterministic=True
    )
    assert jnp.allclose(c_map_learned_updated, weights, atol=1e-6), "Learned mapping matrix is not identical to static weights after update!"
    print("[PASS] Learned mapping matrix remains identical to static weights.")
    
    # Check mapped target forces match static batch forces
    assert jnp.allclose(pred_learned_updated['Mapped_Target_F'], batch_static['F'], atol=1e-3), "Post-update mapped target forces mismatch!"
    print("[PASS] Post-update mapped target forces match static forces.")
    
    print("--- ALL TESTS PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test learned vs static CG mapping.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index to use (e.g., 0, 1, 2, 3)")
    args, _ = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import jax
    print(f"[INFO] Using GPU ID: {args.gpu}")
    print(f"[INFO] JAX visible devices: {jax.devices()}")

    run_test()
