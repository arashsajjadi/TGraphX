import torch
from torch.utils.data import Dataset, DataLoader
from .graph import Graph, GraphBatch

class GraphDataset(Dataset):
    r"""Dataset for graphs.

    Args:
        graphs (list[Graph]): A list of Graph objects.
    """
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

def graph_collate_fn(batch):
    """Default collate function to batch a list of Graph objects."""
    return GraphBatch(batch)

class GraphDataLoader(DataLoader):
    r"""DataLoader for GraphDataset with an optional custom collate function."""
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, **kwargs):
        if collate_fn is None:
            collate_fn = graph_collate_fn
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs
        )
