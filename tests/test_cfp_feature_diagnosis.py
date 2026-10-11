import numpy as np
import importlib.util
from pathlib import Path

# An installed third-party `tools` package shadows the repository namespace.
spec = importlib.util.spec_from_file_location('cfp_diagnostic_cli', Path(__file__).parents[1] / 'tools/cfp_diagnose_features.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
auc, diagnostic_partition = module.auc, module.diagnostic_partition


def test_auc_ties_and_separated_scores():
    assert auc(np.array([1., 1.]), np.array([0., 0.])) == 1.
    assert auc(np.array([1., 1.]), np.array([1., 1.])) == .5


def test_identity_retrieval_excludes_newborns_and_degenerate_fusion_equals_rgb():
    records = []
    features = []
    for time in [1, 26, 51]:
        for identity in range(3):
            records.append(dict(sequence='a', frame_id=time, track_id=identity + 1, partition='dev'))
            features.append(np.eye(3)[identity])
    features = np.array(features, np.float32)
    config = dict(seed=0, minimum_frame_gap=25, pairs_per_class_per_sequence=20)
    result = diagnostic_partition(records, features, features, 'dev', config)['a']
    assert result['eligible_queries'] == 6
    for mode in ['rgb', 'ir', 'zero_ir', 'repeat_rgb', 'equal_fusion']:
        assert result['interventions'][mode]['causal_oracle_gallery_recall1'] == 1.
        assert result['interventions'][mode]['pair_auc'] == 1.
