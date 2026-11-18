"""Descriptive QC plots returning Axes without saving or showing figures."""
import numpy as np
from .api import summarize_qc

def _axes(ax):
    if ax is not None: return ax
    try: import matplotlib.pyplot as plt
    except ImportError as exc: raise ImportError('Install readsparrow[plots] for Matplotlib') from exc
    return plt.subplots()[1]

def _sample(frame,sample):
    summaries=summarize_qc(frame)
    if 'assigned_reads' in summaries:
        if sample is not None and sample!=summaries['sample']: raise ValueError("Requested sample is absent")
        return summaries['sample'],summaries
    if sample is None: raise ValueError("Choose sample= for a multiple-sample DataFrame")
    if sample not in summaries: raise ValueError("Requested sample is absent")
    return sample,summaries[sample]

def abundance_distribution(counts_df,*,sample=None,ax=None,bins=40,**kwargs):
    """Histogram of log2(count+1), annotated with raw-count Gini/median depth."""
    sample,qc=_sample(counts_df,sample); ax=_axes(ax)
    ax.hist(np.log2(counts_df[sample].to_numpy(dtype=float)+1),bins=bins,**kwargs)
    ax.set(xlabel='log2(read count + 1)',ylabel='Reference guides',title=str(sample))
    ax.text(.98,.97,f"Median depth: {qc['median_count']:g}\nRaw-count Gini: {qc['raw_count_gini']:.3f}",
            transform=ax.transAxes,ha='right',va='top')
    return ax

def depth_representation(counts_df,*,sample=None,ax=None,thresholds=(1,10,30,100,300,1000),**kwargs):
    """Reference guides with count >= each minimum read-count threshold (%)."""
    sample,_=_sample(counts_df,sample); thresholds=np.asarray(thresholds,dtype=float)
    if thresholds.ndim!=1 or not len(thresholds) or not np.isfinite(thresholds).all() or np.any(thresholds<=0) or np.any(np.diff(thresholds)<=0):
        raise ValueError("thresholds must be positive finite and strictly increasing")
    values=counts_df[sample].to_numpy(dtype=float); retained=[100*float((values>=t).mean()) for t in thresholds]
    ax=_axes(ax); kwargs.setdefault('marker','o'); ax.plot(thresholds,retained,**kwargs)
    ax.set(xscale='log',xlabel='Minimum read count',ylabel='Reference guides retained (%)',ylim=(0,105),title=str(sample))
    ax.set_xticks(thresholds,labels=[f'{t:g}' for t in thresholds]); return ax

def rejection_breakdown(rejection_stats,*,ax=None,**kwargs):
    """Plot the complete assigned/rejected partition returned by rs.count."""
    if not isinstance(rejection_stats,dict) or not rejection_stats: raise ValueError("Provide category->read counts")
    labels={'assigned_exact':'Assigned exact guide','no_left_anchor':'Missing upstream anchor',
        'no_usable_right_anchor':'Missing usable downstream anchor',
        'anchor_region_no_exact_reference_match':'No exact guide match',
        'ambiguous_multiple_guides':'Multiple guide identities','anchor_span_exceeds_limit':'Anchor span exceeds limit'}
    names=[]; values=[]
    for key,value in rejection_stats.items():
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<0:
            raise ValueError("Category counts must be nonnegative integers")
        names.append(labels.get(key,key.replace('_',' '))); values.append(int(value))
    ax=_axes(ax); bars=ax.barh(names,values,**kwargs); ax.bar_label(bars,padding=3)
    ax.set(xlabel='Reads',title='Assignment and rejection breakdown'); ax.margins(x=.2); return ax
