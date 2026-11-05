"""Exact 3D joint-shell identities and finite-cache MI, not thermalization tests."""

import itertools

import numpy as np
import pytest
import torch

from recurrent_rg.empirical_mi import (
    _region_codes,
    binary_lattice_mi_statistics,
    lattice_mi_statistics,
)
from recurrent_rg.rsmi import kernel_probabilities, region_offsets, sample_regions


def explicit_joint_counts(fields, *, flip):
    """Independent Python arbitrary-integer identities for all 152 shell sites."""
    visible = list(itertools.product(range(2), repeat=3))
    shell = [offset for offset in itertools.product(range(-2, 4), repeat=3)
             if any(position in (-2, 3) for position in offset)]
    results = []
    for ensemble in fields[..., 0].numpy():
        pairs = []
        for sample in ensemble:
            for origin in itertools.product(*(range(n) for n in sample.shape)):
                def code(offsets):
                    return sum(int(sample[tuple((o+d) % n for o,d,n in
                        zip(origin, offset, sample.shape))] > 0) << i
                        for i, offset in enumerate(offsets))
                v,e = code(visible),code(shell)
                pairs.append((v,e))
                if flip:
                    pairs.append((255-v,(1 << 152)-1-e))
        categories = {value: i for i,value in enumerate(sorted({e for _,e in pairs}))}
        counts = torch.zeros(256,len(categories),dtype=torch.double)
        for v,e in pairs:
            counts[v,categories[e]] += 1
        results.append(counts/counts.sum())
    return results


def uncompressed_information(joints, table):
    values = []
    for joint in joints:
        he = table.T @ joint
        independent = he.sum(1,keepdim=True)*he.sum(0,keepdim=True)
        terms = he * (he.clamp_min(1e-300).log()-independent.clamp_min(1e-300).log())
        values.append(terms.sum())
    return torch.stack(values)


def test_3d_full_shell_geometry_translation_and_no_buffer_aliasing():
    v,e = region_offsets(3,block=2,buffer=1)
    assert v.shape == (8,3)
    assert e.shape == (6**3-4**3,3)
    assert len({tuple(offset) for offset in e.tolist()}) == 152
    assert all(any(t in (-2,3) for t in offset) for offset in e.tolist())
    assert not set(map(tuple,v.tolist())) & set(map(tuple,e.tolist()))
    assert torch.cdist(v.double(),e.double(),p=float('inf')).min() == 2
    generator = torch.Generator().manual_seed(36)
    x = torch.arange(2*3*6**3).reshape(2,3,6,6,6,1)
    a,b = sample_regions(x,2,2,generator,buffer=1)
    assert a.shape == (2,2,8) and b.shape == (2,2,152)
    with pytest.raises(ValueError,match='non-aliasing'):
        sample_regions(torch.ones(2,3,5,6,6,1),2,2,generator,buffer=1)


def test_multiword_environment_distinguishes_digits_above_128_bits():
    # At origin zero all four configurations agree in the low 128 shell bits.
    # Their last 24 bits differ; neither int64 truncation nor a two-word scheme
    # may identify them as the same joint environment category.
    _,e = region_offsets(3,block=2,buffer=1)
    colors = torch.zeros(4,6,6,6,dtype=torch.long)
    for sample,index in ((1,129),(2,151),(3,129),(3,151)):
        offset = e[index].tolist()
        colors[(sample,)+tuple(t % 6 for t in offset)] = 1
    codes,cardinalities = _region_codes(colors,e,2)
    assert codes.shape == (4*6**3,3)
    assert cardinalities.tolist() == [2**61,2**61,2**30]
    origin_codes = codes[np.arange(4)*6**3]
    assert np.unique(origin_codes,axis=0).shape[0] == 4
    np.testing.assert_array_equal(origin_codes[:,:2],np.zeros((4,2),dtype=np.int64))
    assert np.all(origin_codes >= 0)


@pytest.mark.parametrize('flip',[False,True])
def test_3d_mi_and_all_256_kernel_gradients_match_arbitrary_integer_joint(flip):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(361)
    fields = (2*torch.randint(2,(2,3,6,6,6,1),generator=generator)-1).double()
    # Repeated and uniform configurations ensure non-singleton environments are
    # present too; this is not merely an all-singleton entropy identity test.
    fields[:,1] = fields[:,0]
    fields[:,2] = -1
    hist = binary_lattice_mi_statistics(fields,block=2,flip=flip)
    assert hist.conditional.is_sparse
    assert hist.conditional.shape[1] == 256
    assert hist.diagnostics['environment_sites'] == 152
    assert hist.diagnostics['environment_identity_encoding'] == 'joint_multiword_base_q'
    assert hist.diagnostics['environment_identity_words'] == 3
    assert hist.diagnostics['environment_projection'] is False
    assert hist.diagnostics['observations_by_ensemble'] == [648*(2 if flip else 1)]*2
    logits = torch.randn(256,2,generator=generator,dtype=torch.double,requires_grad=True)
    table = logits.softmax(-1)
    actual = hist.information(table)
    expected = uncompressed_information(explicit_joint_counts(fields,flip=flip),table)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0],
        torch.autograd.grad(expected.sum(),logits)[0],atol=2e-12,rtol=1e-9)
    hist.prepare_selection()
    torch.testing.assert_close(hist.information_for(table,[1,0,1]),actual[[1,0,1]],
                               atol=2e-12,rtol=1e-10)


def test_3d_sparse_dense_and_translation_invariance_agree():
    generator = torch.Generator().manual_seed(362)
    fields = (2*torch.randint(2,(2,2,6,6,6,1),generator=generator)-1).double()
    sparse = binary_lattice_mi_statistics(fields)
    dense = lattice_mi_statistics(fields,states=2,sparse_conditionals=False)
    translated = binary_lattice_mi_statistics(torch.roll(fields,(1,-2,3),(2,3,4)))
    logits = torch.randn(256,2,generator=generator,dtype=torch.double)
    table = logits.softmax(-1)
    torch.testing.assert_close(sparse.information(table),dense.information(table),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(sparse.information(table),translated.information(table),atol=1e-12,rtol=1e-10)


def test_3d_pointmass_acceleration_also_retains_nondeterministic_conditionals():
    fields = -torch.ones(1,3,6,6,6,1,dtype=torch.double)
    # At origin zero these samples share the complete all-negative shell,
    # while their V codes differ: P(V|E) must not become a point mass there.
    fields[0,1,0,0,0,0] = 1
    fields[0,2,0:2,0:2,0:2,0] = 1
    hist = binary_lattice_mi_statistics(fields)
    conditional = hist.conditional.to_dense()
    assert bool(((conditional > 0).sum(1) > 1).any())
    generator = torch.Generator().manual_seed(366)
    logits = torch.randn(256,2,generator=generator,dtype=torch.double,requires_grad=True)
    table = logits.softmax(-1)
    actual = hist.information(table)
    expected = uncompressed_information(explicit_joint_counts(fields,flip=False),table)
    torch.testing.assert_close(actual,expected,atol=2e-12,rtol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),logits,retain_graph=True)[0],
        torch.autograd.grad(expected.sum(),logits)[0],atol=2e-12,rtol=1e-9)


def test_3d_exact_mi_gradients_reach_existing_binary_c_mlp_without_other_parameters():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Linear(5,1)

    generator = torch.Generator().manual_seed(365)
    fields = (2*torch.randint(2,(1,2,6,6,6,1),generator=generator)-1).double()
    hist = binary_lattice_mi_statistics(fields)
    # Same public C lookup interface as the model, with all 256 blocks, not
    # straight-through or sampled outputs. Unrelated Q parameters stay unused.
    kernel = torch.nn.Module()
    kernel.local_encoder = torch.nn.Sequential(torch.nn.Linear(8,5),torch.nn.Tanh())
    kernel.field_head = Head()
    kernel.double()
    unrelated_q = torch.nn.Linear(8,1).double()
    patterns = (2*((torch.arange(256)[:,None] // (2**torch.arange(8))) % 2)-1).double()
    table = kernel_probabilities(kernel,patterns[None])[0]
    actual = hist.information(table).sum()
    expected = uncompressed_information(explicit_joint_counts(fields,flip=False),table).sum()
    parameters = tuple(kernel.parameters())
    actual_grad = torch.autograd.grad(actual,parameters,retain_graph=True)
    expected_grad = torch.autograd.grad(expected,parameters)
    for left,right in zip(actual_grad,expected_grad):
        assert bool(torch.isfinite(left).all())
        torch.testing.assert_close(left,right,atol=2e-12,rtol=1e-9)
    assert sum(float(gradient.abs().sum()) for gradient in actual_grad) > 0
    assert all(parameter.grad is None for parameter in unrelated_q.parameters())


def test_3d_singleton_shuffled_null_can_be_large_and_is_not_information_proof():
    generator = torch.Generator().manual_seed(363)
    fields = (2*torch.randint(2,(1,2,6,6,6,1),generator=generator)-1).double()
    original = binary_lattice_mi_statistics(fields)
    shuffled = binary_lattice_mi_statistics(fields,shuffle_visible_seed=363)
    # Independent random spins nevertheless give large plug-in MI with an
    # identifying 152-bit shell. Shuffling cannot cure the singleton bias.
    parity = ((torch.arange(256)[:,None] // (2**torch.arange(8))) % 2).sum(1) % 2
    table = torch.stack((1-parity,parity),1).double()
    assert original.diagnostics['mean_fraction_environment_categories_seen_once'] == 1
    assert original.diagnostics['mean_fraction_observations_in_singleton_environment_categories'] == 1
    assert float(shuffled.information(table)[0]) > .65
    torch.testing.assert_close(original.information(table),shuffled.information(table),atol=1e-12,rtol=1e-10)
    assert shuffled.diagnostics['shuffle_visible_seed'] == 363
    assert 'may remain positive' in shuffled.diagnostics['shuffle_null_qualification']
    assert 'No shuffled-null' in original.diagnostics['shuffle_null_qualification']


@pytest.mark.parametrize('dimension,sites',[(1,2),(2,20)])
def test_legacy_region_codes_remain_single_word_exact(dimension,sites):
    generator = torch.Generator().manual_seed(364)
    colors = torch.randint(2,(3,*([6]*dimension)),generator=generator)
    v,e = region_offsets(dimension)
    assert len(e) == sites
    for offsets in (v,e):
        codes,cardinality = _region_codes(colors,offsets,2)
        reference = sum(torch.roll(colors,tuple(-int(t) for t in offset),
                                  tuple(range(1,dimension+1)))*2**i
                        for i,offset in enumerate(offsets)).flatten().numpy()
        np.testing.assert_array_equal(codes,reference)
        assert cardinality.tolist() == [2**len(offsets)]
