import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cfp.online import CurrentFrameFrontend, CurrentFrameRGBTTransform, jpeg_response, identity_ir_view
from yolox.data import RGBTValTransform

def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


online_cli = load_cli('cfp_run_online')
storage_guard = online_cli.storage_guard
select_campaigns = load_cli('cfp_select_predictive_campaign').select_campaigns


def test_jpeg_response_matches_canonical_disk_codec(tmp_path):
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (48, 80, 3), dtype=np.uint8)
    path = tmp_path / 'image.jpg'
    assert cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert np.array_equal(jpeg_response(pixels), cv2.imread(str(path)))


@pytest.mark.parametrize('shape,size', [((17, 31, 3), (24, 40)), ((31, 17, 3), (24, 40)), ((1080, 1920, 3), (800, 1440))])
def test_lookup_preprocessing_is_bit_exact_for_resize_padding_and_all_uint8_levels(shape, size):
    means, std = (.485, .456, .406), (.229, .224, .225)
    ir_means, ir_std = (.0651, .0262, .0581), (.1916, .1015, .1084)
    rng = np.random.default_rng(4)
    rgb, ir = [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(2)]
    original = RGBTValTransform(rgb_means=means, std=std, ir_means=ir_means, ir_std=ir_std)
    optimized = CurrentFrameRGBTTransform(means, std, ir_means, ir_std)
    expected, _ = original(rgb, ir, None, size)
    actual, _ = optimized(rgb, ir, None, size)
    assert actual.flags.c_contiguous and np.array_equal(actual, expected)


def test_current_frame_frontend_never_reads_provided_ir_in_causal_mode():
    frontend = CurrentFrameFrontend.__new__(CurrentFrameFrontend)
    frontend.input_regime = 'causal'
    image = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match='own current/past'):
        frontend.process(image, provided_ir=image)


def test_pixel_interventions_preserve_source_pixels_and_use_current_frame_only():
    rgb = np.full((4, 5, 3), 80, np.uint8)
    ir = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    original = ir.copy()
    assert identity_ir_view(rgb, ir, 'none') is ir
    assert identity_ir_view(rgb, ir, 'repeat_rgb') is rgb
    assert not identity_ir_view(rgb, ir, 'zero').any()
    shuffled = identity_ir_view(rgb, ir, 'spatial_shuffle')
    assert np.array_equal(shuffled, identity_ir_view(rgb, ir, 'spatial_shuffle'))
    assert sorted(map(tuple, shuffled.reshape(-1, 3))) == sorted(map(tuple, ir.reshape(-1, 3)))
    assert np.array_equal(ir, original)
    with pytest.raises(ValueError):
        identity_ir_view(rgb, ir, 'future')


def test_fixed_proposal_frontend_calls_generator_once_and_needs_no_detector():
    frontend = CurrentFrameFrontend.__new__(CurrentFrameFrontend)
    frontend.input_regime = 'causal'
    calls = []
    frontend.generator = lambda image: calls.append(image.copy()) or image
    frontend.rgb_encoder = None
    image = np.full((8, 8, 3), 20, np.uint8)
    boxes, scores = np.array([[1., 1., 6., 6.]], np.float32), np.array([.9], np.float32)
    result = frontend.process(image, fixed_proposals=(boxes, scores))
    assert len(calls) == 1 and np.array_equal(calls[0], image)
    assert result['boxes'] is boxes and result['rgb_features'] is None
    assert 'detector_nms' not in result['stage_seconds']
    with pytest.raises(ValueError, match='Finite fixed'):
        frontend.process(image, fixed_proposals=(boxes * np.nan, scores))


def test_storage_reserve_and_shared_filesystem_growth_stop_before_new_work(monkeypatch):
    monkeypatch.setattr(online_cli.shutil, 'disk_usage', lambda _: SimpleNamespace(free=40 * 2**30))
    assert storage_guard(Path('.'), 30, 40 * 2**30, 2) == 40 * 2**30
    with pytest.raises(OSError, match='reserve/output budget'):
        storage_guard(Path('.'), 45, 40 * 2**30, 2)
    with pytest.raises(OSError, match='reserve/output budget'):
        storage_guard(Path('.'), 30, 43 * 2**30, 2)


def test_selection_uses_mean_over_all_seeds_and_rejects_missing_setting(tmp_path):
    paths = []
    for seed in range(3):
        runs = {}
        for weight in [.1, .3, .5]:
            # A winning seed0 at .1 must not hide failures on the other seeds.
            hota = (90 if seed == 0 else 10) if weight == .1 else (60 if weight == .3 else 50)
            runs[f'trial_transformer_seed{seed}_w{weight}'] = dict(metrics=dict(COMBINED_SEQ=dict(HOTA=hota, AssA=40, IDF1=60, IDSW=86)))
        path = tmp_path / f'seed{seed}.json'
        path.write_text(json.dumps(dict(status='COMPLETE', runs=runs)))
        paths.append(path)
    selected = select_campaigns(paths)['families']['trial_transformer']['selected']
    assert selected['appearance_weight'] == .3 and selected['mean']['HOTA'] == 60
    d = json.loads(paths[-1].read_text())
    del d['runs']['trial_transformer_seed2_w0.5']
    paths[-1].write_text(json.dumps(d))
    with pytest.raises(ValueError, match='Incomplete matched'):
        select_campaigns(paths)
