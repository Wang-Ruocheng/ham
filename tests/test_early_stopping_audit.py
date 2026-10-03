import pytest
from scripts.audit_potts_early_stopping import select_checkpoints


def test_selection_and_online_patience():
    result=select_checkpoints([400,800,1200,1600,2000],[1.,.8,.81,.82,.5],[.01]*5)
    assert result==dict(best_validation=2000,one_standard_error=2000,patience_selected=800,patience_stop=1600)


def test_one_standard_error_prefers_earlier():
    result=select_checkpoints([400,800,1200],[.105,.1,.102],[.01,.01,.01])
    assert result['one_standard_error']==400
    assert result['best_validation']==800


def test_invalid_statistics():
    with pytest.raises(ValueError):select_checkpoints([],[],[])
    with pytest.raises(ValueError):select_checkpoints([400],[float('nan')],[0.])
