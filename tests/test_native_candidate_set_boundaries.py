"""Candidate-set boundary summaries are retrospective and label-explicit."""

import gzip
import json

import numpy as np

from cfp.native_candidate_set_audit import summarize_trace


def test_candidate_set_summary_separates_unknown_and_competition(tmp_path):
    path = tmp_path / "trace.jsonl.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(json.dumps({"frame": 2, "rows": [{
            "history": [[1, 0]], "proposals": [1, 2, 3],
            "raw_geometry": [0.1, 0.2, 0.3],
            "baseline": [0.2, 0.1, 0.4],
        }]}) + "\n")
    physical = {"labels": np.array([4, 4, -1, 9]),
                "frame_of_proposal": np.array([1, 2, 2, 2])}
    result = summarize_trace(path, physical)
    counts = result["counts"]
    assert counts["clean_positive_rows"] == 1
    assert counts["unknown_legal_rows"] == 1
    assert counts["competition_rows"] == 1
    assert counts["clean_positive_unknown_competition_rows"] == 1
    assert counts["clean_baseline_top1_correct"] == 0
