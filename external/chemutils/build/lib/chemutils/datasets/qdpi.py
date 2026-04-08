"""Downloads and prepares the SPICE dataset."""

from urllib import request
from pathlib import Path
from os import walk

import periodictable
import re
import jax
import numpy as onp
import dpdata
import functools
import jaxopt
import jax.numpy as jnp

from jax.scipy.optimize import minimize
from chemtrain.data import preprocessing
from chemutils.datasets import utils

qdpi_versions: dict[str, str] = {
    'v1.0.0': "https://zenodo.org/records/14970869/files/QDpiDataset-main.tar.gz?download=1"
}

def download_qdpi(root="./_data",
                   scale_R=0.1,
                   scale_U=96.4853722,
                   fractional=True,
                   max_samples=None,
                   version='v1.0.0',
                   subsets=None,
                   **kwargs):
    """Download complete SPICE dataset.

    Args:
        root: Download directory of QDpi dataset.
        scale_R: Scaling factor for atomic positions (default=1e-1 #A -> nm).
        scale_U: Scaling factor for potential energies (default=2625.5 #eV -> kJ/mol).
        fractional: Whether to scale positions by simulation box (default=true).
        subsets: List of regex to select subsets of the dataset

    Returns:
        The complete QDpi dataset.
    """

    with jax.default_device(jax.devices("cpu")[0]):
        data_dir = download_source(qdpi_versions[version], root=root)

        dataset = load_and_padd_samples(data_dir)
        dataset, subsets = select_subsets(data_dir, dataset, subsets)
        dataset = scale_dataset(dataset, scale_R, scale_U, fractional)
        dataset = calc_formation_energies(dataset, species_indep_shift=False)
        dataset = split_by_subset(dataset, max_samples, **kwargs)


    info = {
        "subsets": subsets,
        "version": qdpi_versions,
        "scaling": {
            "R": scale_R,
            "U": scale_U,
            "fractional": fractional
        },
    }

    return dataset, info


def download_source(url: str, root: str="./_data"):
    """Downloads and unpacks the QDpi dataset.
    'https://gitlab.com/RutgersLBSR/QDpiDataset/-/tree/main?ref_type=heads',
    'https://zenodo.org/records/14970869'
    give an overview on the available versions.

    Args:
        url: Url for dataset download.
        root: Download directory of QDpi dataset.

    Returns:
        Path of the download directory.
    """
    data_dir = Path(root)
    data_dir.mkdir(exist_ok=True, parents=True)

    if not (data_dir / "QDpi").exists():
        (data_dir / "QDpi").mkdir()
        print(f"Download QDpi dataset from {url}")
        request.urlretrieve(url, data_dir / "QDpi/QDpiDataset-main.tar.gz", utils.show_progress)
        import tarfile
        tar = tarfile.open(data_dir / "QDpi/QDpiDataset-main.tar.gz")
        tar.extractall(path=f"{data_dir}/QDpi")
        tar.close()

    return data_dir / "QDpi"


def load_and_padd_samples(data_dir):
    """Loads and padds the atom data from every subset."""

    # Do not process more than once
    if (data_dir / "QDpi.npz").exists():
        return dict(onp.load(data_dir / "QDpi.npz"))

    subsets = []
    dataset_dict = dict(
        species=[],  # species information in atomic numbers
        coords=[],
        forces=[],
        energies=[],
        subset=[]
    )
    subset_idx = 0
    for (dirpath, _, filenames) in walk(data_dir):
        if not filenames:
            continue
        for filename in filenames:
            if not filename.endswith('.hdf5'):
                continue
            print(f"Process {dirpath}/{filename}")
            subsets.append(f"{dirpath}/{filename}")
            data = dpdata.MultiSystems()
            data.from_deepmd_hdf5(f"{dirpath}/{filename}")
            for frame in list(data.systems.values()):
                atom_types = frame.data['atom_types']
                atom_names = frame.data['atom_names']
                elements = [atom_names[element] for element in atom_types]
                species = [periodictable.elements.symbol(element).number for element in elements]
                dataset_dict["species"].append(species)
                dataset_dict["coords"].append(frame.data['coords'])
                dataset_dict["forces"].append(frame.data['forces'])
                dataset_dict["energies"].append(frame.data['energies'])
                dataset_dict["subset"].append(subset_idx)
            subset_idx += 1

    max_atoms = max([len(species_frame) for species_frame in dataset_dict["species"]])
    n_samples = sum([coords_frame.shape[0] for coords_frame in dataset_dict["coords"]])
    n_mols = len(dataset_dict["energies"])

    print(f"Found {subset_idx} subsets with {n_mols} molecules, a maximum of {max_atoms} atoms and {n_samples} samples.")

    # Reserve memory for the complete padded dataset
    dataset = {
        "id": onp.zeros((n_samples,), dtype=int),
        "R": onp.zeros((n_samples, max_atoms, 3)),
        "F": onp.zeros((n_samples, max_atoms, 3)),
        "U": onp.zeros((n_samples,)),
        "c": onp.zeros((n_samples,), dtype=int),
        "subset": onp.zeros((n_samples,), dtype=int),
        "species": onp.zeros((n_samples, max_atoms), dtype=int),
        "mask": onp.zeros((n_samples, max_atoms), dtype=bool),
    }

    idx = 0
    for id, conf in enumerate(dict(zip(dataset_dict.keys(), values)) for values in zip(*dataset_dict.values())):
        n_atoms = len(conf["species"])
        n_frames = conf["coords"].shape[0]
        dataset["id"][idx:idx + n_frames] = onp.broadcast_to(id, (n_frames,))
        dataset["subset"][idx:idx + n_frames] = onp.broadcast_to(conf["subset"], (n_frames,))
        dataset["mask"][idx:idx + n_frames] = onp.broadcast_to(onp.arange(max_atoms) < n_atoms, (n_frames, max_atoms))
        dataset["species"][idx:idx + n_frames, :n_atoms] = onp.broadcast_to(onp.asarray(conf["species"], dtype=int), (n_frames, n_atoms))

        dataset["R"][idx:idx + n_frames, :n_atoms, :] = onp.asarray(conf["coords"])
        dataset["F"][idx:idx + n_frames, :n_atoms, :] = onp.asarray(conf["forces"])
        dataset["U"][idx:idx + n_frames] = onp.asarray(conf["energies"])

        idx += n_frames

    # save dataset
    onp.savez(data_dir / "QDpi.npz", **dataset)
    with open(data_dir / "subsets.dat", "w") as file:
        file.writelines([f"{str(idx).rjust(3, '0')} {subset}\n"
                         for idx, subset in enumerate(subsets)])

    return dataset


def calc_formation_energies(dataset, species_indep_shift=False, maxiter=1000):
    """Calculate formation energies by subtracting reference energies of individual atoms from total energies. This is
    achieved by minimizing the difference between total energy and per species potential energy shift.
    """
    print("> Calculate formation energies by removing per species reference energies")
    _species = dataset.get("species")
    _total_energy = dataset.get('U')

    # fit per species shift
    _unique_species = jnp.unique(_species)
    _n_species = int(jnp.max(_unique_species)) + 1

    # number of trainable parameters
    n_params = _unique_species.shape[0] - 1
    if species_indep_shift:
        n_params += 1
    # initial parameter set
    x0 = jnp.zeros((n_params,))

    @jax.vmap
    def _params_energy_shift_map(species):
        # map from parameters to all atom types
        _t1 = jnp.transpose(jax.nn.one_hot(_unique_species, _n_species))
        # map from atom types to species fo configuration
        _t2 = jax.nn.one_hot(species, _n_species)
        _t = jnp.matmul(_t2, _t1)
        params_energy_shift_map = jnp.sum(_t, axis=0)
        if species_indep_shift:
            return params_energy_shift_map.at[0].set(1)
        else:
            return params_energy_shift_map[1:] # removing padded atoms (atomic number = 0)

    _params_energy_shift_map = _params_energy_shift_map(_species)
    a = onp.asarray(_params_energy_shift_map)

    def _loss(params, params_energy_shift_map, total_energy):

        @functools.partial(jax.vmap, in_axes=(0, None))
        def _per_species_energy_shift(params_energy_shift_map, params):
            return jnp.vecdot(params_energy_shift_map, params)

        # calculate per species potential shift
        _dU = _per_species_energy_shift(params_energy_shift_map, params)

        return jax.numpy.linalg.norm(total_energy - _dU)

    def _formation_energy_calculation(params, params_energy_shift_map, total_energy):

        @functools.partial(jax.vmap, in_axes=(0, None))
        def _per_species_energy_shift(params_energy_shift_map, params):
            return jnp.vecdot(params_energy_shift_map, params)

        # calculate per species potential shift
        _dU = _per_species_energy_shift(params_energy_shift_map, params)

        return total_energy - _dU

    solver = jaxopt.LBFGS(fun=_loss, maxiter=maxiter)
    res = solver.run(x0, params_energy_shift_map=_params_energy_shift_map, total_energy=_total_energy)
    print("> Finished optimization")

    dataset["formation_energy"] = onp.asarray(_formation_energy_calculation(res.params, _params_energy_shift_map, _total_energy))

    return dataset


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
        dataset[split]['U_weight'] = weights
        dataset[split]['F_weight'] = max_particles / n_particles

    return dataset


def select_subsets(data_dir, dataset, subsets):
    """Selects the subsets from the dataset."""
    if subsets is None:
        return dataset

    with open(data_dir / "subsets.dat", "r") as file:
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


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset from eV to kJ/mol and A to nm."""

    box = 10 * (dataset["R"].max() - dataset["R"].min())

    print(f"Original positions: {dataset['R']}")
    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")


    if fractional:
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * onp.tile(box * onp.eye(3), (dataset['R'].shape[0], 1, 1))
    dataset['U'] *= scale_U
    dataset['F'] *= scale_F

    return dataset


if __name__ == "__main__":
    dataset = download_qdpi(root="/home/paul/Datasets")

    for key in dataset.keys():
        print(f"Split has species in range {dataset[key]['species'].min()} to {dataset[key]['species'].max()}")
        print(f"Unique values are {onp.unique(dataset[key]['species'])}")