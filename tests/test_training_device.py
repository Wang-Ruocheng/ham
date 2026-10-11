"""Execution-device changes must not relax the physics recipe."""
import json
from pathlib import Path
import pytest
import torch

from recurrent_rg.cli import main as cli
from experiments.rsmi_separate_qcr import main, build_parser


def test_default_cpu_and_opt_in_timing():
    args=build_parser().parse_args(['--config','unused','--output','unused'])
    assert args.device=='cpu' and args.benchmark_warmup is None


def test_cli_forwards_cuda_without_ablation(monkeypatch):
    calls=[]
    monkeypatch.setattr('experiments.rsmi_separate_qcr.main',lambda argv:calls.append(argv))
    cli(['train','--config','c','--output','out','--device','cuda:0','--benchmark-warmup','2'])
    assert calls==[['--config','c','--output','out','--device','cuda:0','--benchmark-warmup','2']]


def test_bad_device_and_timing_fail_before_reading_data():
    with pytest.raises(ValueError,match='Supported execution devices'):
        main(['--config','unused','--output','unused','--device','mps'])
    with pytest.raises(ValueError,match='nonnegative'):
        main(['--config','unused','--output','unused','--benchmark-warmup','-1'])


@pytest.mark.skipif(torch.cuda.is_available(),reason='Checks unavailable-CUDA fail-closed behavior')
def test_requested_unavailable_cuda_does_not_fall_back_to_cpu():
    with pytest.raises(RuntimeError,match='unavailable'):
        main(['--config','unused','--output','unused','--device','cuda:0'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA')
def test_cuda_driver_actual_training_reports_and_resume_state(tmp_path):
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/remote_smoke_225.json').read_text())
    gen=torch.Generator().manual_seed(817)
    train=(2*torch.randint(2,(3,8,36,36,1),generator=gen)-1).double()
    test=(2*torch.randint(2,(2,8,36,36,1),generator=gen)-1).double()
    cache=tmp_path/'cache.pt'
    torch.save((train,torch.tensor([.1,.3,.5],dtype=torch.float64),
                test,torch.tensor([.2,.4],dtype=torch.float64)),cache)
    config['data'].update(cache=str(cache))
    config['model']['hidden']=8
    config['training'].update(epochs=4,threads=1,checkpoint_every=2,log_every=2)
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(config))
    whole=tmp_path/'whole';first=tmp_path/'first';resumed=tmp_path/'resumed'
    common=['--config',str(cfg),'--device','cuda:0','--benchmark-warmup','1']
    main(common+['--output',str(whole)])
    main(common+['--output',str(first),'--epochs','2'])
    main(common+['--output',str(resumed),'--resume-checkpoint',str(first/'checkpoint_last.pt')])
    a=torch.load(whole/'checkpoint_last.pt',weights_only=True)
    b=torch.load(resumed/'checkpoint_last.pt',weights_only=True)
    assert {str(v.device) for v in a['model'].values()}=={'cuda:0'}
    assert a['metadata']['execution_device']=='cuda:0'
    assert a['metadata']['independent_c'] and not a['metadata']['labels_used_in_training']
    for key,value in a['model'].items():
        # CUDA index_add_ may use nondeterministic atomic reduction order.
        # Restore Adam/RNG exactly, but do not promise bitwise weight equality.
        torch.testing.assert_close(value,b['model'][key],atol=1e-10,rtol=1e-9)
    assert torch.equal(a['cuda_rng_state'],b['cuda_rng_state'])
    summary=json.loads((whole/'summary.json').read_text())
    assert summary['actual_epochs']==4
    assert summary['timing']['measured_updates']==3
    assert all(value>0 for value in summary['timing']['step_seconds_all'])
    assert (whole/'report'/'R_vs_g.png').is_file()
