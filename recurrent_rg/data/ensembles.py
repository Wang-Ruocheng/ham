"""Data/label boundary, grouped splits, and explicit replica pooling."""
from dataclasses import dataclass
import torch
from ..tensor_io import all_finite, load_cpu_tensors


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
        if not all_finite(x) or any(v is not None and not all_finite(v) for v in (y,self.group_ids)):
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


def _validate_temperature_supervised_disorder_split(train, test, contract):
    """Allow repeated T targets only for an explicitly verified whole-J split.

    Temperature is the supervised target, not the physical-source identity in
    this protocol.  All legacy caches retain the stricter label-disjoint rule.
    This receipt does not certify equilibration or independent replica samples.
    """
    if (not isinstance(contract, dict)
            or contract.get('kind') != 'whole_disorder_temperature_supervision'
            or contract.get('group_columns') != ['disorder_id', 'temperature_index']
            or contract.get('supervision_columns') != ['T']):
        raise ValueError('Invalid whole-disorder temperature-supervision split contract')
    bond_mapping = contract.get('bond_sha256_by_disorder')
    if not isinstance(bond_mapping, dict):
        raise ValueError('Temperature-supervision split requires disorder bond fingerprints')

    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    used_disorders, split_disorders, temperature_by_index = set(), [], {}
    for dataset in (train, test):
        groups, temperatures = dataset.group_ids, dataset.couplings
        if (groups is None or groups.shape != (len(dataset.configurations), 2)
                or groups.dtype not in integer_dtypes or torch.any(groups < 0)):
            raise ValueError('Temperature-supervision groups must be nonnegative integer [J,T_index] rows')
        if (temperatures is None or temperatures.shape != (len(dataset.configurations), 1)
                or temperatures.dtype == torch.bool or temperatures.is_complex()
                or not torch.isfinite(temperatures).all() or torch.any(temperatures <= 0)):
            raise ValueError('Temperature supervision requires exactly one positive finite T column')
        cells, disorders = set(), set()
        for (disorder, index), (temperature,) in zip(groups.tolist(), temperatures.tolist()):
            if (disorder, index) in cells:
                raise ValueError('Duplicate disorder-temperature cell in supervised split')
            cells.add((disorder, index))
            disorders.add(disorder)
            if index in temperature_by_index and temperature_by_index[index] != temperature:
                raise ValueError('Temperature index has inconsistent T targets across disorders')
            temperature_by_index[index] = temperature
        used_disorders.update(disorders)
        split_disorders.append(disorders)
    if split_disorders[0] & split_disorders[1]:
        raise ValueError('Temperature supervision must hold out whole disjoint disorders')
    if set(bond_mapping) != {str(disorder) for disorder in used_disorders}:
        raise ValueError('Bond fingerprints must cover exactly the supervised disorder identities')
    fingerprints = list(bond_mapping.values())
    if any(not isinstance(value, str) or len(value) != 64
           or any(character not in '0123456789abcdef' for character in value)
           for value in fingerprints):
        raise ValueError('Require valid lowercase SHA256 fingerprints for supervised disorder bonds')
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('Identical bonds cannot be assigned different supervised disorder identities')


def load_ensembles(path, *, mmap=False):
    """Read tuple/versioned caches; retain labels outside model/loss batches.

    mmap is execution-only and opt-in for other historical pipeline callers.
    Callers must not mutate a mapped cache; training selects independent copies.
    """
    payload = load_cpu_tensors(path, mmap=mmap)
    if isinstance(payload,dict) and payload.get('schema_version')==1:
        train=EnsembleDataset(**payload['train']).validate()
        test=EnsembleDataset(**payload['test']).validate()
    elif isinstance(payload,(tuple,list)) and len(payload)==4:
        tx, ty, vx, vy = payload
        train, test = EnsembleDataset(tx, ty).validate(), EnsembleDataset(vx, vy).validate()
    else:
        raise ValueError('Expected (train_X, train_labels, test_X, test_labels) cache')
    split_contract = payload.get('split_contract') if isinstance(payload, dict) else None
    shared_temperature_targets = split_contract is not None
    if shared_temperature_targets:
        _validate_temperature_supervised_disorder_split(train, test, split_contract)
    if (train.group_ids is None)!=(test.group_ids is None):
        raise ValueError('Both splits must use the same group-ID convention')
    a = {tuple(row) for row in train.groups().tolist()}
    b = {tuple(row) for row in test.groups().tolist()}
    if a & b:
        raise ValueError('Train/test parameter cells overlap')
    if train.couplings is not None and test.couplings is not None and not shared_temperature_targets:
        a={tuple(row) for row in train.couplings.reshape(len(train.configurations),-1).tolist()}
        b={tuple(row) for row in test.couplings.reshape(len(test.configurations),-1).tolist()}
        if a & b: raise ValueError('Train/test parameter cells overlap despite different group IDs')
    if train.configurations.shape[2:] != test.configurations.shape[2:]:
        raise ValueError('Native train/test cache geometries differ; evaluate another size explicitly')
    return train, test
