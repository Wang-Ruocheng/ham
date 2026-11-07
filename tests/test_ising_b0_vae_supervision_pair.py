import json
import os
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments import rsmi_separate_qcr as driver
from recurrent_rg.config import RunConfig
from recurrent_rg.checkpoints import build_model
from recurrent_rg.data.statistics import plaquette_distribution
from recurrent_rg.losses import qcr_objective
from scripts.report_ising_b0_supervision_pair import scalar_flow, main as report_main
from scripts.run_ising_b0_supervision_pair import commands


CONFIG='configs/ising2d_B0_weak_kl_vae_g1_supervision_pair_seed91_20261106.json'


def test_explicit_vae_supervision_validation_and_paired_commands(tmp_path):
    for weight in [0,10]:
        a=SimpleNamespace(q_coupling_weight=weight,q_future_weight=0,independent_c=True)
        driver.validate_ensemble_vae_training(a)
        a.independent_c=False
        with pytest.raises(ValueError):driver.validate_ensemble_vae_training(a)
        a.independent_c=True;a.q_future_weight=1
        with pytest.raises(ValueError):driver.validate_ensemble_vae_training(a)
    stages=commands(dict(config=CONFIG,mi_cache_dir='cache'),tmp_path)
    assert [x[0] for x in stages]==['unsupervised','supervised','report']
    assert stages[0][1][stages[0][1].index('--q-coupling-weight')+1]=='0'
    assert stages[1][1][stages[1][1].index('--q-coupling-weight')+1]=='10'


def test_supervision_changes_Q_not_C_R_gradients_or_random_stream():
    torch.set_num_threads(1);c=RunConfig.load(CONFIG);torch.manual_seed(91)
    m=build_model(c.model,2);m.detach_kernel_representation=True
    x=(2*torch.randint(2,(2,4,36,36,1))-1).double()
    labels=torch.tensor([[.3],[.6]],dtype=torch.double,requires_grad=True)
    results=[]
    for weight in [0,10]:
        torch.manual_seed(55)
        total,t=qcr_objective(m,x,pattern_fn=plaquette_distribution,**c.loss.objective_kwargs())
        before=torch.get_rng_state().clone()
        sup=driver.microscopic_coupling_loss(m.theory_encoder,x,labels)
        assert torch.equal(before,torch.get_rng_state())
        driver.independent_component_gradients(total+weight*sup,t['closure_total'],
            2*t['kernel_noncollapse']+t['kernel_sign'],m.theory_encoder.parameters(),m.rg_kernel.parameters(),m.theory_rg.parameters())
        results.append([[p.grad.clone() for p in mod.parameters()] for mod in [m.theory_encoder,m.rg_kernel,m.theory_rg]])
    assert any(not torch.equal(a,b) for a,b in zip(results[0][0],results[1][0]))
    for i in [1,2]:
        for a,b in zip(results[0][i],results[1][i]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert labels.grad is None


def test_scalar_search_keeps_all_crossings_and_support():
    class Model:
        def advance_theory(self,x):return x+.2*(x+.5)*(x-.3)
    direct=np.linspace(-1,1,12).reshape(4,3,1);labels=np.arange(4.)
    _,_,roots,search=scalar_flow(Model(),direct,labels,3)
    assert [r['g'] for r in roots]==pytest.approx([-.5,.3])
    assert [r['stability'] for r in roots]==['attracting','repelling']
    assert search['thermal_nu'] is None


def test_real_paired_cpu_driver_and_complete_report(tmp_path,monkeypatch):
    torch.set_num_threads(1);gen=torch.Generator().manual_seed(20)
    x=(2*torch.randint(2,(6,64,36,36,1),generator=gen)-1).double()
    data=tmp_path/'data.pt';torch.save((x[:3],torch.tensor([.2,.4,.6]),x[3:],torch.tensor([.3,.5,.7])),data)
    c=RunConfig.load(CONFIG)
    c=replace(c,data=replace(c.data,cache=str(data)),
        training=replace(c.training,epochs=2,system_batch=2,sample_batch=4,log_every=1,checkpoint_every=1,
                         terminal_r_max_steps=2,terminal_r_cache_repeats=1),
        evaluation=replace(c.evaluation,samples=4,systems=2))
    config=tmp_path/'config.json';config.write_text(json.dumps(c.to_dict()))
    device=os.environ.get('QCR_PAIR_SMOKE_DEVICE','cpu')
    extra=['--mi-cache-dir',str(tmp_path/'mi_cache'),'--mi-test-cpu'] if device.startswith('cuda') else []
    for name,weight in [('unsupervised',0),('supervised',10)]:
        driver.main(['--config',str(config),'--output',str(tmp_path/name),'--q-coupling-weight',str(weight),
                     '--skip-final-report','--device',device]+extra)
        manifest=json.loads((tmp_path/name/'manifest.json').read_text())
        assert manifest['labels_used_in_training']==bool(weight)
        assert (tmp_path/name/'r_only/summary.json').exists()
    monkeypatch.setattr('sys.argv',['report','--root',str(tmp_path)])
    report_main()
    report=json.loads((tmp_path/'report/report.json').read_text())
    assert report['status']=='complete' and report['paired_initialization']
    assert report['C_comparison']['max_absolute_parameter_difference']<1e-10
    assert (tmp_path/'report/R_flow.png').exists()
