from datasets import Dataset


def get_subset(dset: Dataset, percent_to_save: float) -> Dataset:
    subset = range(int(len(dset) * percent_to_save))
    dset = dset.select(subset)
    dset = dset.flatten_indices()
    return dset
