import torch
from recurrent_rg.model_nd import RecurrentRGNetwork2D


def test_resume_training_preserves_adam_and_batch_sequence(tmp_path):
    from recurrent_rg.model_nd import RecurrentRGNetwork1D
    from experiments.joint_qcr_ising import train, two_spin_distribution
    torch.manual_seed(91)
    model = RecurrentRGNetwork1D(levels=2, hidden=4, theory_dim=2).double()
    x = (2 * torch.randint(2, (4, 4, 8, 1)) - 1).double()
    kwargs = dict(pattern_fn=two_spin_distribution, system_batch=3,
                  sample_batch=4, eval_samples=4, eval_systems=3,
                  learning_rate=.001, seed=91,
                  objective_kwargs={'objective_mode': 'data_geometry',
                                    'coordinate_reduction': 'sum'})
    history = train(model, x, x, epochs=501, progress_dir=tmp_path, **kwargs)
    state = torch.load(tmp_path / 'training_step_500.pt', weights_only=True)
    resumed = RecurrentRGNetwork1D(levels=2, hidden=4, theory_dim=2).double()
    resumed_history = train(resumed, x, x, epochs=501, resume_state=state,
                            initial_history=[r for r in history if r['epoch'] <= 500], **kwargs)
    for name, tensor in model.state_dict().items():
        if not name.startswith('action_network.'):
            torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
    assert history == resumed_history


def test_checkpoint_encoder_matches_values_and_input_parameter_gradients():
    torch.manual_seed(11)
    model = RecurrentRGNetwork2D(levels=3, hidden=8, theory_dim=2).double()
    x = torch.randn(3, 4, 8, 8, 1, dtype=torch.float64, requires_grad=True)
    parameters = tuple(model.theory_encoder.parameters())
    first = model.theory_encoder(x)
    grad = torch.autograd.grad(first.square().sum(), (x,)+parameters)
    model.theory_encoder.checkpoint_chunks = 1
    second = model.theory_encoder(x)
    other = torch.autograd.grad(second.square().sum(), (x,)+parameters)
    torch.testing.assert_close(first, second)
    for a, b in zip(grad, other):
        torch.testing.assert_close(a, b)


def test_binary_lookup_matches_entire_qcr_loss_and_gradients():
    from experiments.joint_qcr_ising import qcr_objective, plaquette_distribution
    from experiments._utils import balanced_views
    torch.manual_seed(12)
    model = RecurrentRGNetwork2D(levels=3, hidden=8, theory_dim=2).double()
    # A nonconstant C avoids nondifferentiable zero-distance/max ties in L_sep.
    with torch.no_grad():
        for p in model.rg_kernel.parameters():
            p.zero_()
        model.rg_kernel.local_encoder[0].weight[0, 0] = 1.
        model.rg_kernel.local_encoder[2].weight[0, 0] = 1.
        model.rg_kernel.field_head.projection.weight[0, 0] = 1.
    x = (2*torch.randint(2, (4, 6, 16, 16, 1))-1).double()
    x = balanced_views(x, 6)
    parameters = tuple(p for mod in (model.rg_kernel, model.theory_encoder, model.theory_rg)
                       for p in mod.parameters())
    loss, _ = qcr_objective(model, x, pattern_fn=plaquette_distribution)
    original = torch.autograd.grad(loss, parameters)
    model.theory_encoder.binary_lookup = True
    model.rg_kernel.binary_lookup = True
    other, _ = qcr_objective(model, x, pattern_fn=plaquette_distribution)
    gradients = torch.autograd.grad(other, parameters)
    torch.testing.assert_close(loss, other, atol=1e-10, rtol=1e-8)
    for a, b in zip(original, gradients):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
