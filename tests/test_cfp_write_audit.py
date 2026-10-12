import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('cfp_write_audit_cli',Path(__file__).parents[1]/'tools/cfp_audit_identity_writes.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_refused_observation_never_reanchors_or_counts_as_written():
    records=[dict(frame=i+1,track_id=1,proposal_index=i,written=written,accepted=written) for i,written in enumerate([True,False,True,True,False])]
    result=module.audit_records(records,np.array([3,7,7,3,7]),np.arange(1,6),{})
    assert result['counts']==dict(records=5,actually_written=3,known_anchor_initializations=1,refused_or_unavailable=2,refused_wrong=2,written_wrong=1,written_correct=1)
    assert result['existing_known_write_precision']==.5
    assert result['refused_known_wrong_fraction']==1.
    assert result['examples'][0]['original_past_physical_anchor']==3
    ambiguous=module.audit_records(records,np.array([3,7,7,3,7]),np.arange(1,6),{3:{7}})
    assert ambiguous['counts']['unknown_current_physical_label']==1 and ambiguous['existing_known_write_precision']==1.


def test_current_proposal_frame_and_unique_write_guard():
    record=dict(frame=1,track_id=1,proposal_index=0,written=True,accepted=True)
    with pytest.raises(ValueError):module.audit_records([record,record],np.array([3]),np.array([1]),{})
    with pytest.raises(ValueError):module.audit_records([record],np.array([3]),np.array([2]),{})
