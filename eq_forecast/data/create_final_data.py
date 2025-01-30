from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split


def create_datasets(dataset):

    kwargs = {"y_pad":dataset.pad_labels}

    dataset = StaticGraphTemporalSignal(
        edge_index = dataset.edge_index,
        edge_weight = dataset.edge_attr,
        features = dataset.feature_mats,
        targets = dataset.labels,
        **kwargs
    )

    train_dataset, test_val_dataset = temporal_signal_split(dataset, train_ratio=0.8)

    test_dataset, val_dataset = temporal_signal_split(test_val_dataset, train_ratio=0.5)


    return train_dataset, val_dataset, test_dataset