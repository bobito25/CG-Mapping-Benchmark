"""Downloads and prepares the SPICE dataset."""
import subprocess
from urllib import request
from pathlib import Path

from os.path import join as opj

import h5py

import re
import jax
import numpy as onp

from chemtrain.data import preprocessing
from chemutils.datasets import utils

_url = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/"
_source = "mdcath_source.h5"

def download_mdCATH(root="./_data",
                    scale_R=0.0529177,
                    scale_U=2625.5,
                    fractional=True,
                    max_samples=None,
                    version='v2.0.1',
                    subsets=None,
                    **kwargs):
    """Download complete SPICE dataset.

    Args:
        root: Download directory of SPICE dataset.
        scale_R: Scaling factor for atomic positions (default=5.2917721e-2 #Bohr -> nm).
        scale_U: Scaling factor for potential energies (default=2625.5 #Hartee -> kJ/mol).
        fractional: Whether to scale positions by simulation box (default=true).
        url: Url for dataset download (default refers to SPICE v2.0.1).
        subsets: List of regex to select subsets of the dataset

    Returns:
        The complete SPICE dataset.
    """

    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(_url, root=root)

    #     dataset = load_and_padd_samples(data_dir)
    #     dataset, subsets = select_subsets(data_dir, dataset, subsets)
    #     dataset = scale_dataset(dataset, scale_R, scale_U, fractional)
    #     dataset = split_by_subset(dataset, max_samples, **kwargs)
    #
    #
    # info = {
    #     "subsets": subsets,
    #     "version": spice_versions,
    #     "scaling": {
    #         "R": scale_R,
    #         "U": scale_U,
    #         "fractional": fractional
    #     },
    # }
    #
    # return dataset, info


def download_source(url: str, root: str="./_data"):
    """Downloads the mdCATH dataset (3TB).

    Args:
        url: Url for dataset download.
        root: Download directory of SPICE dataset.

    Returns:
        Path of the download directory.
    """
    data_dir = Path(root) / "mdCATH"
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir / _source).exists():
        print(f"Download mainfile {_source} from {opj(url, _source)}")
        request.urlretrieve(opj(url, _source), data_dir / _source, utils.show_progress)

    # Iterate over the source file and download the individual files
    with h5py.File(data_dir / _source, "r") as file:
        for idx, pdb_id in enumerate(file.keys()):
            df = data_dir / f"mdcath_dataset_{pdb_id}.h5"
            try:
                h5py.File(df, "r")
            except:
                print(f"({idx + 1}/{len(file.keys())}) Download {opj(url, f'data/mdcath_dataset_{pdb_id}.h5')}")
                request.urlretrieve(opj(url, f"data/mdcath_dataset_{pdb_id}.h5"), df, utils.show_progress)

    return data_dir


def prepare_gromacs_solvation(data_dir, pdb_id):

    out_dir = data_dir / "computations" / pdb_id
    out_dir.mkdir(exist_ok=True, parents=True)
    out_dir.absolute()

    # Create a gromacs input file from the topology
    with h5py.File(data_dir / f"mdcath_dataset_{pdb_id}.h5", "r") as file:
        with open(out_dir / "solvated.pdb", "w") as pdb:
            pdb.write(
                file[pdb_id]["pdb"][()].decode('utf-8')
                # .replace(" HSD ", " HID ")
                .replace("CAY", "CB ")
                .replace("HY1", "HA ")
                .replace("HY2", "HA ")
                .replace("HY3", "HA ")
                .replace("CY ", "C  ")
                .replace("OY ", "O  ")
                .replace("NT ", "N  ")
                .replace("HNT", "H  ")
                .replace("CAT", "CA ")
                .replace("HT1", "HA ")
                .replace("HT2", "HA ")
                .replace("HT3", "HA ")

                # TODO: Check renaming
                # .replace("CAT VAL", "CH3 VAL")
                # .replace("HT1 VAL", "HA1 VAL")
                # .replace("HT2 VAL", "HA2 VAL")
                # .replace("HT3 VAL", "HA3 VAL")
                # .replace("TIP3W", "SOL")
                # .replace("SOD  SOD", "Na   SOL")
                # .replace("CLA  CLA", "Cl   SOL")
                # .replace("OH", "OH1")
            )

    # Convert into a gromacs topology
    subprocess.call(f"cd {out_dir} && {GMX_PATH} pdb2gmx -f solvated.pdb -o solvated.gro -chainsep ter -ff charmm22star -water tip3p", shell=True)




# def load_and_padd_samples(data_dir):
#     """Loads and padds the atom data."""
#
#     # Do not process more than once
#     if (data_dir / "SPICE.npz").exists():
#         return dict(onp.load(data_dir / "SPICE.npz"))
#
#     with h5py.File(data_dir / "SPICE.hdf5", "r") as file:
#         subsets = []
#         for mol in file.keys():
#             current_subset = file[mol]["subset"][0]
#             if not current_subset in subsets:
#                 print(f"Discovered new subset: {current_subset}")
#                 subsets.append(current_subset)
#
#         max_atoms = max([file[mol]["atomic_numbers"].size for mol in file.keys()])
#         n_samples = sum([file[mol]["conformations"].shape[0] for mol in file.keys()])
#
#         mols = list(file.keys())
#         mols.sort()
#
#         print(f"Found {len(file.keys())} molecules with a maximum of {max_atoms} atoms and {n_samples} samples.")
#
#         # Reserve memory for the complete padded dataset
#         dataset = {
#             "id": onp.zeros((n_samples,), dtype=int),
#             "R": onp.zeros((n_samples, max_atoms, 3)),
#             "F": onp.zeros((n_samples, max_atoms, 3)),
#             "U": onp.zeros((n_samples,)),
#             "c": onp.zeros((n_samples,), dtype=int),
#             "subset": onp.zeros((n_samples,), dtype=int),
#             "species": onp.zeros((n_samples, max_atoms), dtype=int),
#             "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
#         }
#
#         idx = 0
#         for id, mol in enumerate(mols):
#             confs = file[mol]
#             conf_shape = confs["conformations"].shape
#             if onp.size(conf_shape)==3:
#                 n_samples, n_atoms, _ = conf_shape
#             else:
#                 # skip damaged molecule (no conformations, forces, energies, ...)
#                 continue
#
#             dataset["id"][idx:idx + n_samples] = onp.broadcast_to(id, (n_samples,))
#             dataset["subset"][idx:idx + n_samples] = onp.broadcast_to(subsets.index(confs["subset"][0]), (n_samples,))
#             dataset["mask"][idx:idx + n_samples] = onp.broadcast_to(onp.arange(max_atoms) < n_atoms, (n_samples, max_atoms))
#             dataset["species"][idx:idx + n_samples, :n_atoms] = onp.broadcast_to(onp.asarray(confs["atomic_numbers"], dtype=int), (n_samples, n_atoms))
#
#             dataset["R"][idx:idx + n_samples, :n_atoms, :] = onp.asarray(confs["conformations"])
#             dataset["F"][idx:idx + n_samples, :n_atoms, :] = -1.0 * onp.asarray(confs["dft_total_gradient"])
#             dataset["U"][idx:idx + n_samples] = onp.asarray(confs["formation_energy"])
#
#             idx += n_samples
#
#     # save dataset
#     onp.savez(data_dir / "SPICE.npz", **dataset)
#     with open(data_dir / "subsets.dat", "w") as file:
#         file.writelines([f"{str(idx).rjust(3, '0')} {subset.decode('ascii')}\n"
#                          for idx, subset in enumerate(subsets)])
#
#     return dataset
#
#
# def split_by_subset(dataset, max_samples=None, **kwargs):
#     """Splits the loaded subsets individually."""
#
#     # Find out which subsets were loaded
#     total_samples = dataset["subset"].size
#     subsets = onp.unique(dataset["subset"])
#     keys = dataset.keys()
#     split_dataset = {
#         "training": [], "validation": [], "testing": []
#     }
#
#
#     for subset in subsets:
#         sub_dataset = {key: arr[dataset["subset"] == subset] for key, arr in dataset.items()}
#         sub_train, sub_val, sub_test = preprocessing.train_val_test_split(sub_dataset, **kwargs, shuffle=True, shuffle_seed=11)
#
#         # Select a maximum number of samples relative to the total number of samples
#         if max_samples is not None:
#             max_subset = int(sub_dataset["subset"].size / total_samples * max_samples)
#             sub_train = {key: arr[:max_subset] for key, arr in sub_train.items()}
#             sub_val = {key: arr[:max_subset] for key, arr in sub_val.items()}
#             sub_test = {key: arr[:max_subset] for key, arr in sub_test.items()}
#
#         split_dataset["training"].append(sub_train)
#         split_dataset["validation"].append(sub_val)
#         split_dataset["testing"].append(sub_test)
#
#
#     # Concatenate the subsets
#     final_dataset = {}
#     for split, split_data in split_dataset.items():
#         final_dataset[split] = {
#             key: onp.concatenate([sub[key] for sub in split_data], axis=0)
#             for key in keys
#         }
#
#     return final_dataset
#
#
# def process_dataset(dataset):
#     """Creates weights for masked loss."""
#
#     for split in dataset.keys():
#         # Weight the potential by the number of particles. The per-particle
#         # potential should have equal weight, so the error for larger systems
#         # should contribute more. On the other hand, since we compute the
#         # MSE for the forces, we have to correct for systems with masked
#         # out particles.
#
#         n_particles = onp.sum(dataset[split]['mask'], axis=1)
#         max_particles = dataset[split]['mask'].shape[1]
#
#         weights = n_particles / onp.mean(n_particles, keepdims=True)
#         dataset[split]['U_weight'] = weights
#         dataset[split]['F_weight'] = max_particles / n_particles
#
#     return dataset
#
#
# def select_subsets(data_dir, dataset, subsets):
#     """Selects the subsets from the dataset."""
#     if subsets is None:
#         return dataset
#
#     with open(data_dir / "subsets.dat", "r") as file:
#         selection: dict[int, str] = {
#             int(line.partition(" ")[0]): line.partition(" ")[2]
#             for line in file.readlines()
#             if any([re.search(s, line) for s in subsets])
#         }
#
#     boolean_mask = onp.isin(dataset["subset"], list(selection.keys()))
#     dataset = {
#         key: arr[boolean_mask] for key, arr in dataset.items()
#     }
#
#     return dataset, {idx: sel.strip("\n") for idx, sel in selection.items()}
#
#
# def scale_dataset(dataset, scale_R, scale_U, fractional=True):
#     """Scales the dataset from Hartee to kJ/mol and Bohr to nm."""
#
#     box = 10 * (dataset["R"].max() - dataset["R"].min())
#
#     print(f"Original positions: {dataset['R']}")
#     print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")
#
#
#     if fractional:
#         dataset['R'] = dataset['R'] / box
#     else:
#         dataset['R'] = dataset['R'] * scale_R
#
#     print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")
#
#     scale_F = scale_U / scale_R
#     dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
#     dataset['U'] *= scale_U
#
#     # Remove a species energy term
#
#     dataset['F'] *= scale_F
#
#     return dataset
#
#
if __name__ == "__main__":
    GMX_PATH = "/usr/local/gromacs/bin/gmx"

    # dataset = download_mdCATH(root="/media/HardDrive/Datasets")
    prepare_gromacs_solvation(Path("/media/HardDrive/Datasets/mdCATH"), "4mb4A02")

