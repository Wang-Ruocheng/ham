from types import SimpleNamespace
import json
from pathlib import Path

import pytest
import torch

from experiments.rsmi_separate_qcr import selected_mi_execution, validate_training_clock, build_parser, main


def test_selected_mi_route_preserves_historical_exact_resume():
    assert selected_mi_execution({'enabled':True}, True, None)
    assert not selected_mi_execution({'enabled':False}, True, None)
    assert not selected_mi_execution({'enabled':True}, False, None)
    assert not selected_mi_execution({'enabled':True}, True, {'metadata':{}})
    assert not selected_mi_execution({'enabled':True}, True,
        {'metadata':{'execution_optimizations':{'selected_empirical_mi':False}}})
    assert selected_mi_execution({'enabled':True}, True,
        {'metadata':{'execution_optimizations':{'selected_empirical_mi':True}}})


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_invalid_training_clock(value):
    with pytest.raises(ValueError,match='finite and positive'):
        validate_training_clock(SimpleNamespace(max_training_seconds=value,
                                               max_wall_seconds=None,source=None))


def test_training_clock_is_explicit_and_cannot_change_old_wall_scope():
    args = build_parser().parse_args(['--config','c','--output','o','--max-training-seconds','900'])
    validate_training_clock(args)
    assert args.max_training_seconds == 900 and args.max_wall_seconds is None
    args.max_wall_seconds = 1200
    with pytest.raises(ValueError,match='not both'):
        validate_training_clock(args)
    args.max_wall_seconds, args.source = None, 'historical'
    with pytest.raises(ValueError,match='standalone'):
        validate_training_clock(args)


def test_training_only_cap_records_actual_updates_without_reducing_config(tmp_path):
    raw = json.loads((Path(__file__).resolve().parents[1] / 'configs/remote_smoke_225.json').read_text())
    generator = torch.Generator().manual_seed(33)
    train = (2 * torch.randint(2, (3, 8, 36, 36, 1), generator=generator) - 1).double()
    test = (2 * torch.randint(2, (2, 8, 36, 36, 1), generator=generator) - 1).double()
    cache = tmp_path / 'cache.pt'
    torch.save((train, torch.tensor([.2,.3,.4],dtype=torch.double),
                test, torch.tensor([.5,.6],dtype=torch.double)), cache)
    raw['data']['cache'] = str(cache)
    raw['model']['hidden'] = 5
    raw['training'].update(epochs=8,threads=1,checkpoint_every=2,log_every=2)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(raw))
    run = tmp_path / 'run'
    main(['--config',str(config),'--output',str(run),'--max-training-seconds','1e-12'])
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['actual_epochs'] == 1 and summary['requested_epochs'] == 8
    assert summary['config']['training']['epochs'] == 8
    assert summary['stopped_by_training_limit'] and not summary['stopped_by_wall_limit']
    assert summary['timing']['setup_seconds'] > 0
    checkpoint = torch.load(run / 'checkpoint_last.pt',map_location='cpu',weights_only=True)
    assert checkpoint['epoch'] == 1
