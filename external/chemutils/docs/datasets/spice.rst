SPICE Dataset
=============

The following snippet loads and preprocesses the SPICE dataset.

.. code:
    dataset, info = spice.download_spice(
        "/home/paul/Datasets",
        subsets=config["dataset"].get("subsets"),
        max_samples=config["dataset"].get("max_samples"),
        fractional=False
    )
    dataset = spice.process_dataset(dataset)


.. currentmodule:: chemutils.datasets.spice

.. autofunction:: download_spice

.. autofunction:: process_dataset
