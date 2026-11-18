"""Release portability: packaged versions and independent tiny fixture answers."""
import json
from pathlib import Path
import pandas as pd
import pytest
import readsparrow as rs

BASE=Path(__file__).resolve().parents[1]/'examples/tiny'

@pytest.mark.parametrize('name',['strict_forward','strict_reverse','auto_forward','auto_reverse','mixed','tolerant'])
def test_published_tiny_fixture(name):
    case=json.loads((BASE/'manifest.json').read_text())[name]
    r=rs.count(BASE/case['fastq'],BASE/'reference.tsv',orientation=case['orientation'],anchor_mismatch=case['anchor_mismatch'])
    pd.testing.assert_frame_equal(r.counts_df,pd.read_csv(BASE/'expected'/(name+'_counts.tsv'),sep='\t'))
    assert r.rejection_stats==json.loads((BASE/'expected'/(name+'_rejections.json')).read_text())
    expected=json.loads((BASE/'expected'/(name+'_qc.json')).read_text())
    for key,value in expected.items():
        assert r.qc_summary[key]==pytest.approx(value) if isinstance(value,float) else r.qc_summary[key]==value

def test_stable_runtime_version():
    assert rs.__version__=='0.12.1'
