import torch

from yolox.models.structural_relative_repair import StructuralRepair, relative_semantics, structural_weights


def test_relative_semantics_cancels_common_auxiliary_bias_and_never_changes_geometry():
    torch.manual_seed(4)
    teacher = torch.randn(2, 7, 6)
    current = torch.randn_like(teacher, requires_grad=True)
    null = torch.randn_like(teacher, requires_grad=True)
    output = relative_semantics(current, null, teacher)
    bias = torch.randn_like(teacher)
    shifted = relative_semantics(current+bias, null+bias, teacher)
    torch.testing.assert_close(output, shifted, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output[:, :, :4], teacher[:, :, :4], atol=0, rtol=0)
    assert (output[:, :, 4:]-teacher[:, :, 4:]).abs().max() <= 2
    torch.testing.assert_close(relative_semantics(current, current, teacher), teacher, atol=0, rtol=0)
    output.sum().backward()
    assert current.grad[:, :, :4].count_nonzero() == null.grad[:, :, :4].count_nonzero() == 0
    assert current.grad[:, :, 4:].abs().sum() > 0 and null.grad[:, :, 4:].abs().sum() > 0


def test_structural_inverse_consistency_recovers_a_known_local_translation():
    torch.manual_seed(1)
    query = torch.randn(1, 8, 12, 14)
    key = torch.roll(query, (1, -1), (-2, -1))
    valid = torch.ones(1, 1, 12, 14, dtype=torch.bool)
    weights, confidence = structural_weights(query, key, valid, valid, temperature=.001)
    best = weights.argmax(1).reshape(1, 12, 14)
    expected = (1+2)*5+(-1+2)
    assert (best[:, 3:-3, 3:-3] == expected).all()
    assert (confidence[:, :, 3:-3, 3:-3] > .999).all()
    _, unavailable = structural_weights(query, key, valid & False, valid)
    assert unavailable.count_nonzero() == 0


def test_zero_output_and_missing_auxiliary_return_exact_RGB_for_every_variant():
    torch.manual_seed(3)
    rgb = [torch.randn(2, 4, 8, 10)]
    auxiliary = [torch.randn_like(rgb[0])]
    scores = [torch.rand(2, 1, 8, 10)]
    structure = torch.rand(2, 8, 8, 10); valid = torch.ones(2, 1, 8, 10, dtype=torch.bool)
    counts = []
    for variant in ['full', 'no_structure', 'no_cycle', 'no_centering']:
        model = StructuralRepair(channels=(4,), latent=3, variant=variant)
        counts.append(sum(p.numel() for p in model.parameters()))
        for null in [False, True]:
            output = model(rgb, auxiliary, scores, structure, structure, valid, valid, null=null)
            torch.testing.assert_close(output[0], rgb[0], atol=0, rtol=0)
            output = model(rgb, None, scores, None, None, None, None, availability=False, null=null)
            assert output[0] is rgb[0]
    assert len(set(counts)) == 1


def test_structural_readout_is_independent_of_other_samples():
    torch.manual_seed(7)
    model = StructuralRepair(channels=(4,), latent=3)
    torch.nn.init.normal_(model.levels[0].output.weight, std=.2)
    rgb = torch.randn(2, 4, 8, 10); aux = torch.randn_like(rgb)
    scores = torch.zeros(2, 1, 8, 10)
    q = torch.rand(2, 8, 8, 10); k = torch.rand_like(q)
    valid = torch.ones(2, 1, 8, 10, dtype=torch.bool)
    expected = model([rgb[:1]], [aux[:1]], [scores[:1]], q[:1], k[:1], valid[:1], valid[:1])[0]
    rgb[1] *= 100; aux[1] *= -100; q[1] += 10; k[1] -= 10
    actual = model([rgb], [aux], [scores], q, k, valid, valid)[0][:1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_original_semantic_supervision_trains_the_adapter_through_a_frozen_head():
    from yolox.models.yolo_head import YOLOXHead
    from yolox.models.frozen_head_repair import head_predictions, original_supervised_loss
    torch.manual_seed(13)
    head = YOLOXHead(1, width=.25, in_channels=[32, 64, 128]).eval().requires_grad_(False)
    rgb = [torch.randn(2, c, h, w) for c, h, w in [(8, 16, 24), (16, 8, 12), (32, 4, 6)]]
    auxiliary = [torch.randn_like(value) for value in rgb]
    model = StructuralRepair(channels=(8, 16, 32), latent=4)
    q = torch.rand(2, 8, 16, 24); k = torch.roll(q, (1, -1), (-2, -1))
    valid = torch.ones(2, 1, 16, 24, dtype=torch.bool)
    reference, meta = head_predictions(head, rgb)
    current = head_predictions(head, model(rgb, auxiliary, meta['scores'], q, k, valid, valid))[0]
    null = head_predictions(head, model(rgb, auxiliary, meta['scores'], q, k, valid, valid, null=True))[0]
    prediction = relative_semantics(current, null, reference)
    torch.testing.assert_close(prediction, reference, atol=0, rtol=0)
    targets = torch.zeros(2, 4, 5); targets[:, 0] = torch.tensor([0., 56., 44., 20., 15.])
    parts = original_supervised_loss(head, prediction, meta, targets, (128, 192))
    loss = parts[2]+parts[3]; loss.backward()
    assert torch.isfinite(loss)
    assert sum(layer.output.weight.grad.abs().sum() for layer in model.levels) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in head.parameters())
