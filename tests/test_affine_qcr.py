from dataclasses import replace
import torch

from experiments.rsmi_separate_qcr import independent_component_gradients, build_parser
from recurrent_rg.checkpoints import build_model, qcr_state_dict, load_model_state
from recurrent_rg.config import ModelConfig
from recurrent_rg.data.statistics import categorical_plaquette_distribution
from recurrent_rg.losses import qcr_objective
from recurrent_rg.representation import normalize_affine_decoder_, initialize_affine_decoder_pca_
from recurrent_rg.training import separate_qc_r_gradients


def config():
    return ModelConfig(dimension=2, block_factor=3, categorical_states=3,
                       kernel_sampling='categorical', hidden=5,affine_reconstruction=False)


def test_affine_decoder_preserves_initial_qcr_and_rng_and_roundtrips():
    torch.set_num_threads(1)
    torch.manual_seed(91); a=build_model(config(),1); rng=torch.get_rng_state()
    torch.manual_seed(91); b=build_model(replace(config(),affine_reconstruction=True),1)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in qcr_state_dict(a).items():torch.testing.assert_close(v,qcr_state_dict(b)[k],atol=0,rtol=0)
    decoder=b.theory_encoder.distribution_decoder
    assert isinstance(decoder,torch.nn.Linear) and decoder.out_features==81
    torch.testing.assert_close(decoder.weight.norm(dim=0),torch.ones(1,dtype=torch.double))
    with torch.no_grad():decoder.weight.mul_(3)
    normalize_affine_decoder_(b.theory_encoder)
    torch.testing.assert_close(decoder.weight.norm(dim=0),torch.ones(1,dtype=torch.double))
    c=build_model(replace(config(),affine_reconstruction=True),1)
    load_model_state(c,qcr_state_dict(b))
    for k,v in qcr_state_dict(b).items():torch.testing.assert_close(v,qcr_state_dict(c)[k],atol=0,rtol=0)


def test_affine_objective_updates_decoder_but_c_and_r_gradients_unchanged():
    torch.set_num_threads(1); torch.manual_seed(19)
    x=torch.nn.functional.one_hot(torch.randint(3,(3,8,9,9)),3).double()
    gradients=[]
    for affine in [False,True]:
        torch.manual_seed(91);m=build_model(replace(config(),affine_reconstruction=affine),1)
        m.detach_kernel_representation=True
        torch.manual_seed(31)
        total,terms=qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
            kernel_noncollapse_mode='variance_floor_ensemble',kernel_variance_ratio=1.,kernel_sign_weight=1.)
        closure=5*(terms['commutator']+terms['rollout'])
        c_objective=2*terms['kernel_noncollapse']+terms['kernel_sign']
        independent_component_gradients(total,closure,c_objective,
            m.theory_encoder.parameters(),m.rg_kernel.parameters(),m.theory_rg.parameters())
        gradients.append([[p.grad.clone() for p in module.parameters()] for module in [m.rg_kernel,m.theory_rg]])
        if affine:
            assert 'reconstruction_mse_level_0' in terms and 'reconstruction_js_level_0' not in terms
            assert m.theory_encoder.distribution_decoder.weight.grad.abs().sum()>0
    for ga,gb in zip(*gradients):
        for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_optional_wall_limit_is_not_enabled_by_default():
    parser=build_parser()
    args=parser.parse_args(['--config','x.json','--output','out'])
    assert args.max_wall_seconds is None
    assert args.freeze_c_checkpoint is None and not args.affine_pca_init


def test_pca_initialization_uses_histograms_and_does_not_change_q_c_r_or_rng():
    torch.manual_seed(13)
    m=build_model(replace(config(),affine_reconstruction=True),1)
    before={k:v.clone() for k,v in qcr_state_dict(m).items()}
    probabilities=torch.rand(24,81,dtype=torch.double)
    probabilities/=probabilities.sum(-1,keepdim=True)
    rng=torch.get_rng_state()
    result=initialize_affine_decoder_pca_(m.theory_encoder,probabilities)
    assert not result['labels_used'] and result['decoder_remains_trainable']
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in before.items():
        if 'distribution_decoder' not in k:
            torch.testing.assert_close(v,qcr_state_dict(m)[k],atol=0,rtol=0)
    decoder=m.theory_encoder.distribution_decoder
    torch.testing.assert_close(decoder.bias,probabilities.mean(0))
    torch.testing.assert_close(decoder.weight.norm(),torch.tensor(1.,dtype=torch.double))
    _,_,vh=torch.linalg.svd(probabilities-probabilities.mean(0),full_matrices=False)
    torch.testing.assert_close(abs(decoder.weight[:,0]@vh[0]),torch.tensor(1.,dtype=torch.double))
    assert all(p.requires_grad for p in decoder.parameters())


def test_frozen_c_qr_step_changes_q_r_but_not_c():
    torch.set_num_threads(1);torch.manual_seed(22)
    m=build_model(replace(config(),affine_reconstruction=True),1)
    m.detach_kernel_representation=True
    for p in m.rg_kernel.parameters():p.requires_grad_(False)
    before={k:v.clone() for k,v in qcr_state_dict(m).items()}
    q=list(m.theory_encoder.parameters());r=list(m.theory_rg.parameters())
    opt=torch.optim.Adam(q+r,lr=.001)
    x=torch.nn.functional.one_hot(torch.randint(3,(3,8,9,9)),3).double()
    total,terms=qcr_objective(m,x,pattern_fn=categorical_plaquette_distribution,
        kernel_noncollapse_mode='variance_floor_ensemble',kernel_variance_ratio=1.,kernel_sign_weight=1.)
    closure=5*(terms['commutator']+terms['rollout'])
    separate_qc_r_gradients(total,closure,q,r)
    opt.step();normalize_affine_decoder_(m.theory_encoder)
    for k,v in before.items():
        if k.startswith('rg_kernel.'):
            torch.testing.assert_close(v,qcr_state_dict(m)[k],atol=0,rtol=0)
    assert all(p.grad is None for p in m.rg_kernel.parameters())
    assert any(not torch.equal(before[k],v) for k,v in qcr_state_dict(m).items() if k.startswith('theory_encoder.'))
    assert any(not torch.equal(before[k],v) for k,v in qcr_state_dict(m).items() if k.startswith('theory_rg.'))
