"""Data/label boundary, grouped splits, and explicit replica pooling."""
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class EnsembleDataset:
    configurations: torch.Tensor
    couplings: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None

    def groups(self):
        groups = self.group_ids if self.group_ids is not None else self.couplings
        if groups is None:
            raise ValueError('Need ensemble-source group IDs (or optional couplings) for splitting/pooling')
        return groups.reshape(len(self.configurations), -1)

    def validate(self):
        x, y = self.configurations, self.couplings
        if x.ndim < 4:
            raise ValueError('Expected X[ensembles,samples,*spatial,channels] and matching labels')
        for values in (y,self.group_ids):
            if values is not None and (values.ndim not in (1,2) or len(values)!=len(x)):
                raise ValueError('Metadata must match ensemble count')
        if len(x) < 2 or x.shape[1] < 2 or any(n < 1 for n in x.shape[2:]):
            raise ValueError('Dataset is empty or too small')
        if not torch.isfinite(x).all() or any(v is not None and not torch.isfinite(v).all() for v in (y,self.group_ids)):
            raise ValueError('Dataset contains nonfinite values')
        return self

    def pool(self, replicas_per_ensemble):
        """Pool within an already split dataset; never join spatial sites."""
        self.validate()
        if not isinstance(replicas_per_ensemble, int) or replicas_per_ensemble < 1:
            raise ValueError('Pooling factor must be a positive integer')
        labels = self.groups()
        unique, inverse = torch.unique(labels, dim=0, return_inverse=True)
        xs, ys, gs = [], [], []
        for i in range(len(unique)):
            ids = torch.where(inverse == i)[0]
            if len(ids) % replicas_per_ensemble:
                raise ValueError('Replica count must be divisible by pooling factor at every coupling')
            for group in ids.split(replicas_per_ensemble):
                if self.couplings is not None and not torch.all(self.couplings[group]==self.couplings[group[0]]):
                    raise ValueError('A source group contains conflicting coupling metadata')
                xs.append(self.configurations[group].flatten(0, 1))
                if self.couplings is not None: ys.append(self.couplings[group[0]])
                if self.group_ids is not None: gs.append(self.group_ids[group[0]])
        return EnsembleDataset(torch.stack(xs),torch.stack(ys) if ys else None,torch.stack(gs) if gs else None).validate()


def grouped_split(dataset, *, test_every=3, test_offset=1):
    """Hold out complete parameter cells; no replica leakage."""
    dataset.validate()
    if test_every < 2 or not 0 <= test_offset < test_every:
        raise ValueError('Invalid split rule')
    _, inverse = torch.unique(dataset.groups(), dim=0, return_inverse=True)
    test = inverse % test_every == test_offset
    return tuple(EnsembleDataset(dataset.configurations[mask],
                                dataset.couplings[mask] if dataset.couplings is not None else None,
                                dataset.group_ids[mask] if dataset.group_ids is not None else None).validate()
                 for mask in (~test, test))


def load_ensembles(path):
    """Read legacy tuple caches; retain labels outside model/loss batches."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if isinstance(payload,dict) and payload.get('schema_version')==1:
        train=EnsembleDataset(**payload['train']).validate()
        test=EnsembleDataset(**payload['test']).validate()
    elif isinstance(payload,(tuple,list)) and len(payload)==4:
        tx, ty, vx, vy = payload
        train, test = EnsembleDataset(tx, ty).validate(), EnsembleDataset(vx, vy).validate()
    else:
        raise ValueError('Expected (train_X, train_labels, test_X, test_labels) cache')
    if (train.group_ids is None)!=(test.group_ids is None):
        raise ValueError('Both splits must use the same group-ID convention')
    a = {tuple(row) for row in train.groups().tolist()}
    b = {tuple(row) for row in test.groups().tolist()}
    if a & b:
        raise ValueError('Train/test parameter cells overlap')
    if train.couplings is not None and test.couplings is not None:
        a={tuple(row) for row in train.couplings.reshape(len(train.configurations),-1).tolist()}
        b={tuple(row) for row in test.couplings.reshape(len(test.configurations),-1).tolist()}
        if a & b: raise ValueError('Train/test parameter cells overlap despite different group IDs')
    if train.configurations.shape[2:] != test.configurations.shape[2:]:
        raise ValueError('Native train/test cache geometries differ; evaluate another size explicitly')
    return train, test
