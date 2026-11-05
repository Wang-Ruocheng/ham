"""An explicit whole-J protocol, not a broad bypass of label leakage checks."""
from copy import deepcopy

import pytest
import torch

from recurrent_rg.data.ensembles import load_ensembles


def payload_fixture():
    def split(disorders):
        return dict(configurations=torch.ones((4, 4, 4, 4, 4, 1), dtype=torch.int8),
                    couplings=torch.tensor([[.75], [1.5], [.75], [1.5]], dtype=torch.float64),
                    group_ids=torch.tensor([[disorder, index] for disorder in disorders
                                            for index in (0, 1)], dtype=torch.long))
    return dict(schema_version=1, train=split((0, 2)), test=split((1, 3)),
                split_contract=dict(kind='whole_disorder_temperature_supervision',
                    group_columns=['disorder_id', 'temperature_index'], supervision_columns=['T'],
                    bond_sha256_by_disorder={str(index): character * 64
                                            for index, character in enumerate('abcd')}))


def save_and_load(tmp_path, payload):
    cache = tmp_path / 'ensembles.pt'
    torch.save(payload, cache)
    return load_ensembles(cache)


def test_default_guard_still_rejects_shared_labels_despite_disjoint_groups(tmp_path):
    payload = payload_fixture()
    payload.pop('split_contract')
    with pytest.raises(ValueError, match='despite different group IDs'):
        save_and_load(tmp_path, payload)


def test_complete_whole_disorder_contract_allows_shared_temperature_targets(tmp_path):
    payload = payload_fixture()
    train, test = save_and_load(tmp_path, payload)
    for name, actual in (('train', train), ('test', test)):
        for field in ('configurations', 'couplings', 'group_ids'):
            torch.testing.assert_close(getattr(actual, field), payload[name][field])
    assert set(train.group_ids[:, 0].tolist()).isdisjoint(test.group_ids[:, 0].tolist())
    assert train.couplings.tolist() == test.couplings.tolist()


@pytest.mark.parametrize('change', [
    'kind_only', 'wrong_kind', 'wrong_group_columns', 'wrong_label_columns',
    'missing_bonds', 'extra_bond', 'integer_bond_key', 'duplicate_bonds', 'malformed_sha',
    'uppercase_sha', 'missing_groups', 'one_column_groups', 'three_column_groups',
    'float_groups', 'noninteger_groups', 'bool_groups', 'negative_group',
    'overlapping_J', 'duplicate_cell', 'missing_labels', 'flat_labels',
    'two_label_columns', 'negative_T', 'zero_T', 'nan_T', 'infinite_T', 'bool_T',
    'conflicting_temperature_index',
])
def test_malformed_or_leaking_temperature_contract_is_rejected(tmp_path, change):
    payload = payload_fixture()
    contract = payload['split_contract']
    if change == 'kind_only':
        payload['split_contract'] = dict(kind=contract['kind'])
    elif change == 'wrong_kind':
        contract['kind'] = 'arbitrary_disjoint_groups'
    elif change == 'wrong_group_columns':
        contract['group_columns'] = ['temperature_index', 'disorder_id']
    elif change == 'wrong_label_columns':
        contract['supervision_columns'] = ['beta']
    elif change == 'missing_bonds':
        contract['bond_sha256_by_disorder'].pop('3')
    elif change == 'extra_bond':
        contract['bond_sha256_by_disorder']['4'] = 'e' * 64
    elif change == 'integer_bond_key':
        contract['bond_sha256_by_disorder'][0] = contract['bond_sha256_by_disorder'].pop('0')
    elif change == 'duplicate_bonds':
        contract['bond_sha256_by_disorder']['1'] = contract['bond_sha256_by_disorder']['0']
    elif change == 'malformed_sha':
        contract['bond_sha256_by_disorder']['1'] = 'missing'
    elif change == 'uppercase_sha':
        contract['bond_sha256_by_disorder']['1'] = 'B' * 64
    elif change == 'missing_groups':
        payload['test']['group_ids'] = None
    elif change == 'one_column_groups':
        payload['test']['group_ids'] = payload['test']['group_ids'][:, :1]
    elif change == 'three_column_groups':
        groups = payload['test']['group_ids']
        payload['test']['group_ids'] = torch.cat((groups, groups[:, :1]), 1)
    elif change in ('float_groups', 'noninteger_groups'):
        payload['test']['group_ids'] = payload['test']['group_ids'].double()
        if change == 'noninteger_groups':
            payload['test']['group_ids'][0, 0] = .5
    elif change == 'bool_groups':
        payload['test']['group_ids'] = payload['test']['group_ids'].bool()
    elif change == 'negative_group':
        payload['test']['group_ids'][0, 0] = -1
    elif change == 'overlapping_J':
        payload['test']['group_ids'][:2, 0] = 0
    elif change == 'duplicate_cell':
        payload['test']['group_ids'][1] = payload['test']['group_ids'][0]
        payload['test']['couplings'][1] = payload['test']['couplings'][0]
    elif change == 'missing_labels':
        payload['test']['couplings'] = None
    elif change == 'flat_labels':
        payload['test']['couplings'] = payload['test']['couplings'][:, 0]
    elif change == 'two_label_columns':
        payload['test']['couplings'] = payload['test']['couplings'].repeat(1, 2)
    elif change in ('negative_T', 'zero_T', 'nan_T', 'infinite_T'):
        payload['test']['couplings'][0, 0] = {
            'negative_T': -1., 'zero_T': 0., 'nan_T': float('nan'),
            'infinite_T': float('inf')}[change]
    elif change == 'bool_T':
        payload['test']['couplings'] = payload['test']['couplings'].bool()
    elif change == 'conflicting_temperature_index':
        payload['test']['couplings'][0, 0] = 1.
    else:
        raise AssertionError(change)
    with pytest.raises(ValueError):
        save_and_load(tmp_path, payload)


def test_default_distinct_labels_and_legacy_tuple_stay_compatible(tmp_path):
    payload = payload_fixture()
    payload.pop('split_contract')
    payload['test']['couplings'] += 2.
    train, test = save_and_load(tmp_path, payload)
    assert len(train.configurations) == len(test.configurations) == 4
    legacy = (payload['train']['configurations'], payload['train']['couplings'],
              payload['test']['configurations'], payload['test']['couplings'])
    train, test = save_and_load(tmp_path, legacy)
    assert train.group_ids is test.group_ids is None


def test_loader_does_not_mutate_the_explicit_contract(tmp_path):
    payload = payload_fixture()
    before = deepcopy(payload)
    save_and_load(tmp_path, payload)
    assert payload['split_contract'] == before['split_contract']
    for split in ('train', 'test'):
        for name, value in payload[split].items():
            torch.testing.assert_close(value, before[split][name])
