from experiments.q_original_plateau import plateau_reached


def test_plateau_requires_low_lr_and_full_window():
    rows=[dict(epoch=2400+250*i,best_train=1-.0001*i,lr=1e-6) for i in range(9)]
    assert not plateau_reached(rows,.001)
    assert not plateau_reached(rows[:8],1e-6)
    assert plateau_reached(rows,1e-6)
    rows[0]['lr']=.001
    assert not plateau_reached(rows,1e-6)
    rows[0]['lr']=1e-6
    rows[-1]['best_train']=.9
    assert not plateau_reached(rows,1e-6)
