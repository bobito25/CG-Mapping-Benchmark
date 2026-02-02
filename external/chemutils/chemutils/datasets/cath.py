"""Downloads and prepares the CATH dataset."""

from pathlib import Path

import re
import jax
import numpy as onp

from chemtrain.data import preprocessing
import mdtraj as md
import json
import os

def load_cath(data_dir="./_data",
                scale_R=1,
                scale_U=1,
                fractional=True,
                max_samples=None,
                subsets=None,
                debug=False,
                **kwargs):
    """Load and process Pepsol dataset into workspace.

    Units of Pepsol dataset:
        [R] = nm,
        [U] = kJ/mol,
        [F] = kJ/mol/nm

    Args:
        data_dir: Directory of Pepsol dataset.
        scale_R: Scaling factor for atomic positions (default=1 #nm -> nm).
        fractional: Whether to scale positions by simulation box (default=true).
        subsets: List of regex to select subsets of the dataset.

    Returns:
        The complete Pepsol dataset.
    """
    with jax.default_device(jax.devices("cpu")[0]):
        dataset = load_from_file(data_dir)
        
        if debug:
            print("Debug mode: Limiting each subset to 100 samples.")
            max_per_subset = 100

            # Prepare a new, empty dataset with the same keys
            new_dataset = {key: [] for key in dataset}

            # Iterate over each unique subset label
            for subset_label in set(dataset['subset']):
                # Find the first up to max_per_subset indices for this label
                indices = [i for i, s in enumerate(dataset['subset']) if s == subset_label][:max_per_subset]

                # Copy over those samples for every key
                for key in dataset:
                    for i in indices:
                        new_dataset[key].append(dataset[key][i])

            # Convert lists back to numpy arrays to maintain expected data structure
            for key in new_dataset:
                new_dataset[key] = onp.array(new_dataset[key])

            # Select only subset 0
            print("Selecting only subset [0,1] for debug mode.")
            boolean_mask = onp.isin(new_dataset["subset"], [0,1])
            new_dataset = {key: arr[boolean_mask] for key, arr in new_dataset.items()}

            # Replace the old dataset
            dataset = new_dataset
        
        dataset, cath_add_info, residue_maps = add_residue_info(dataset, debug=debug)
        dataset, subsets = select_subsets(data_dir, dataset, subsets)
        dataset = scale_dataset(dataset, scale_R, scale_U, fractional)
        dataset = split_by_subset(dataset, max_samples, **kwargs)

    info = {
        "subsets": subsets,
        "scaling": {
            "R": scale_R,
            "U": scale_U,
            "fractional": fractional
        },
    }

    return dataset, info


def load_from_file(data_dir, filename="cath.npz"):
    """Load Pepsol dataset from file into workspace

        Args:
            data_dir: Directory of Pepsol dataset.
            filename: Filename of Pepsol dataset.

        Returns:
            The complete Pepsol dataset.
        """
    # Do not process more than once
    assert (Path(data_dir + "/" + filename)).exists(), f'File {data_dir + "/" + filename} does not exist.'
    return dict(onp.load(data_dir + "/" + filename))


def select_subsets(data_dir, dataset, subsets):
    """Selects the subsets from the dataset."""
    if subsets is None:
        return dataset, subsets

    with open(Path(data_dir) / "subsets.dat", "r") as file:
        selection: dict[int, str] = {
            int(line.partition(" ")[0]): line.partition(" ")[2]
            for line in file.readlines()
            if any([re.search(s, line) for s in subsets])
        }

    boolean_mask = onp.isin(dataset["subset"], list(selection.keys()))
    dataset = {
        key: arr[boolean_mask] for key, arr in dataset.items()
    }

    return dataset, {idx: sel.strip("\n") for idx, sel in selection.items()}


def split_by_subset(dataset, max_samples=None, **kwargs):
    """Splits the loaded subsets individually."""

    # Find out which subsets were loaded
    total_samples = dataset["subset"].size
    subsets = onp.unique(dataset["subset"])
    keys = dataset.keys()
    split_dataset = {
        "training": [], "validation": [], "testing": []
    }


    for subset in subsets:
        sub_dataset = {key: arr[dataset["subset"] == subset] for key, arr in dataset.items()}
        sub_train, sub_val, sub_test = preprocessing.train_val_test_split(sub_dataset, **kwargs, shuffle=True, shuffle_seed=11)

        # Select a maximum number of samples relative to the total number of samples
        if max_samples is not None:
            max_subset = int(sub_dataset["subset"].size / total_samples * max_samples)
            sub_train = {key: arr[:max_subset] for key, arr in sub_train.items()}
            sub_val = {key: arr[:max_subset] for key, arr in sub_val.items()}
            sub_test = {key: arr[:max_subset] for key, arr in sub_test.items()}

        split_dataset["training"].append(sub_train)
        split_dataset["validation"].append(sub_val)
        split_dataset["testing"].append(sub_test)


    # Concatenate the subsets
    final_dataset = {}
    for split, split_data in split_dataset.items():
        final_dataset[split] = {
            key: onp.concatenate([sub[key] for sub in split_data], axis=0)
            for key in keys
        }

    return final_dataset


def process_dataset(dataset):
    """Creates weights for masked loss."""

    for split in dataset.keys():
        # Weight the potential by the number of particles. The per-particle
        # potential should have equal weight, so the error for larger systems
        # should contribute more. On the other hand, since we compute the
        # MSE for the forces, we have to correct for systems with masked
        # out particles.

        n_particles = onp.sum(dataset[split]['mask'], axis=1)
        max_particles = dataset[split]['mask'].shape[1]

        weights = n_particles / onp.mean(n_particles, keepdims=True)
        dataset[split]['F_weight'] = max_particles / n_particles

    return dataset


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm."""

    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")

    if fractional:
        box = dataset['box'][0, 0, 0]
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * dataset['box']
    dataset['F'] *= scale_F

    return dataset

def get_gro_path_CATH(subset_key, base_path="/mnt/drives/sdb/jan/Datasets/CATH/simulations"):
    subsets = load_subset_dict()
    code = subsets[subset_key]

    gro_path = f"{base_path}/{code}/md.gro"
    if not os.path.exists(gro_path):
        raise FileNotFoundError(f"{gro_path} does not exist")
    
    return gro_path
    
def add_residue_info(dataset, debug=False):
    subsets = load_subset_dict()
    
    gro_base_path = f"/mnt/drives/sdb/jan/Datasets/CATH/simulations"
    cath_add_info = {}
    residue_maps = {}

    def check_res_map_consistency(residue, atomic_numbers, res_map):
        map_atomic_numbers = res_map[residue]['atomic_numbers']
        for i, atom_num in enumerate(atomic_numbers):
            if atom_num != map_atomic_numbers[i]:
                print(f"  ❌  Residue {residue} mismatch: {atom_num} != {map_atomic_numbers[i]} at index {i}")
                return False
        return True

    pad_length = dataset['species'].shape[1]

    for key, code in subsets.items():
        if debug and key not in ["0", "1"]:
            continue
        print(f"\Adding additionally info for CATH domain {code} (subset_key={key})")

        # Find first sample from training set where the key matches
        mask = dataset['subset'] == int(key)
        cath_species = dataset['species'][mask][0]
        max_length = int(onp.count_nonzero(cath_species))

        gro_path = f"{gro_base_path}/{code}/md.gro"
        if not os.path.exists(gro_path):
            raise FileNotFoundError(f"{gro_path} does not exist")
        
        mdtraj_top = md.load(gro_path).topology        
        cath_add_info[code] = {}
        atomic_numbers = []
        symbols = []
        residue_names = []
        residue_ids = []

        per_residue_atomic_numbers = []
        per_residue_symbols = []
        last = None

        length = 0
        for atom in mdtraj_top.atoms:
            if not atom.residue.is_protein:
                continue
            length += 1
            
            cath_atomic_number = cath_species[atom.index]
            mdtraj_symbol = atom.element.symbol
            mdtraj_number = atom.element.number
            
            if mdtraj_number != cath_atomic_number:
                raise ValueError(f"Atom {atom.index} mismatch: {cath_atomic_number} != {mdtraj_number}")

            atomic_numbers.append(mdtraj_number)
            symbols.append(mdtraj_symbol)
            
            residue_names.append(atom.residue.name)
            residue_ids.append(atom.residue.resSeq)

            if last is None or last != atom.residue.name:
                if last is not None:
                    # Store the residue information
                    residue_maps[last] = {
                        'atomic_numbers': per_residue_atomic_numbers,
                        'symbols': per_residue_symbols,
                        'three-letter-code': last,
                    }
                    
                    # Check consistency
                    if not check_res_map_consistency(last, per_residue_atomic_numbers, residue_maps):
                        raise ValueError(f"Inconsistent residue map for {last} at code {code} with {len(per_residue_atomic_numbers)} atoms.")
                # Reset
                per_residue_atomic_numbers = [mdtraj_number]
                per_residue_symbols = [mdtraj_symbol]
                last = atom.residue.name
            else:
                per_residue_atomic_numbers.append(mdtraj_number)
                per_residue_symbols.append(mdtraj_symbol)
                
        assert length == max_length, f"Length mismatch: {length} != {max_length} for code {code}"
                
        cath_add_info[code]['atomic_numbers'] = atomic_numbers
        cath_add_info[code]['symbols'] = symbols
        cath_add_info[code]['residue_names'] = residue_names
        cath_add_info[code]['residue_ids'] = residue_ids
        
    residue_names_list = []
    residue_ids_list = []
    for subset in dataset['subset']:
        residue_names = cath_add_info[subsets[str(int(subset))]]['residue_names']
        residue_ids = cath_add_info[subsets[str(int(subset))]]['residue_ids']
        
        if len(residue_names) < pad_length:
            # Pad with empty strings or a placeholder
            residue_names = residue_names + [''] * (pad_length - len(residue_names))
        if len(residue_ids) < pad_length:
            # Pad with zeros
            residue_ids = residue_ids + [0] * (pad_length - len(residue_ids))
            
        residue_names_list.append(residue_names)
        residue_ids_list.append(residue_ids)
    
    dataset['residue_names'] = onp.array(residue_names_list)
    # dataset['residue_ids'] = onp.array(residue_ids_list)
    
    return dataset, cath_add_info, residue_maps        
        
def load_subset_dict():
    subsets_path = "/mnt/drives/sda/jan/Datasets/CATH/datasets/v021_half/subsets.dat"
    subsets = {}
    with open(subsets_path, 'r') as f:
        for line in f:
            if line == "\n":
                continue
            # split into ["000", "CATH", "1b43A02"]
            idx_str, _, code = line.strip().split()
            # convert "000" → 0 → "0", "010" → 10 → "10", etc.
            key = str(int(idx_str))
            subsets[key] = code
            
            if code == '4npsA01':
                subsets[key] = "4npsA02"  # typo
    
    assert subsets["0"] == "1b43A02" # first entry is 1b43A02
    return subsets