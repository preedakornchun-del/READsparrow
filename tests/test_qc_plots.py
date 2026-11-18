import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import readsparrow as rs

def test_qc_arithmetic_and_unknown_denominator():
    df=pd.DataFrame({'sgRNA':['a','b','c','d'],'count':[0,1,3,4]})
    q=rs.summarize_qc(df)
    assert q['mapping_rate_percent'] is None
    q=rs.summarize_qc(df,total_reads=10)
    assert q['mapping_rate_percent']==80 and q['representation_percent']==75
    assert q['median_count']==2 and q['mean_count']==2 and q['zero_guides']==1
    x=np.array([0.,1.,3.,4.])
    for values,key in [(x,'raw_count_gini'),(np.log1p(x),'log1p_count_gini')]:
        expected=sum(abs(a-b) for a in values for b in values)/(2*len(x)*values.sum())
        assert q[key]==pytest.approx(expected)
    assert rs.summarize_qc(pd.DataFrame({'count':[0,0]}))['raw_count_gini']==0

@pytest.mark.parametrize('values',[[1,-1],[1,.5],[1,float('nan')],[1,float('inf')],['a','b'],[True,False]])
def test_qc_reject_invalid_counts(values):
    with pytest.raises(ValueError): rs.summarize_qc(pd.DataFrame({'count':values}))

def test_qc_multisample():
    df=pd.DataFrame({'A':[0,1],'B':[2,3]})
    q=rs.summarize_qc(df,total_reads={'A':2,'B':10})
    assert q['A']['mapping_rate_percent']==50 and q['B']['mapping_rate_percent']==50
    with pytest.raises(ValueError): rs.summarize_qc(df,total_reads=10)
    with pytest.raises(ValueError): rs.summarize_qc(df,total_reads={'B':1})
    with pytest.raises(ValueError): rs.plots.abundance_distribution(df)

def test_all_plots_axes_and_data():
    df=pd.DataFrame({'count':[0,1,10,30,100,300,1000]})
    fig,axes=plt.subplots(1,3)
    assert rs.plots.abundance_distribution(df,ax=axes[0]) is axes[0]
    assert rs.plots.depth_representation(df,ax=axes[1]) is axes[1]
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(),np.array([6,5,4,3,2,1])*100/7)
    assert rs.plots.rejection_breakdown({'assigned_exact':6,'no_left_anchor':2},ax=axes[2]) is axes[2]
    assert isinstance(rs.plots.abundance_distribution(df),Axes)
    with pytest.raises(ValueError): rs.plots.depth_representation(df,thresholds=[10,1])
    with pytest.raises(ValueError): rs.plots.rejection_breakdown({'bad':-1})
    plt.close('all')
