"""DataFrame API. Reads are streamed; counts use reference-sized memory."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .counting import Library, load_library
from .orientation import count_extended
from .qc import gini

@dataclass
class CountResult:
    counts_df: object
    qc_summary: dict
    rejection_stats: dict

    def __iter__(self):
        yield self.counts_df
        yield self.qc_summary
        yield self.rejection_stats

def _reference(reference):
    if isinstance(reference, Library): return reference
    if isinstance(reference, (str, Path)): return load_library(reference)
    import pandas as pd
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a path, DataFrame or readsparrow.counting.Library")
    if not reference.columns.is_unique: raise ValueError("Reference column names must be unique")
    choices=[("sgRNA","ID","guide_id"),("Gene","gene_symbol","gene"),("Sequence","seq","sequence")]
    cols=[next((c for c in names if c in reference),None) for names in choices]
    if any(c is None for c in cols) or reference[cols].isna().any().any():
        raise ValueError("Reference needs nonmissing ID, gene and sequence columns")
    try:
        return Library(reference[cols[0]].astype(str).tolist(),reference[cols[1]].astype(str).tolist(),
                       [str(s).upper().encode('ascii') for s in reference[cols[2]]])
    except UnicodeEncodeError as exc: raise ValueError("Reference sequences must be ASCII DNA") from exc

def count(fastq_path,reference,orientation="auto",anchor_mismatch=0,*,sample="count",
          left_anchor="CACCG",right_anchor="GTTTT",max_reads=None,policy="legacy",max_extra=32):
    """Return CountResult(counts_df, qc_summary, rejection_stats) from one FASTQ.

    Exact explicit forward/reverse uses the unchanged baseline loop. Auto scans
    at most20,000 records then reopens/counts once. Mixed attempts forward first.
    anchor_mismatch=1 permits substitutions in each anchor, never in guides.
    A reference DataFrame uses ID/gene/sequence aliases accepted by the CLI.
    """
    if not isinstance(sample,str) or not sample or sample in {"sgRNA","Gene","Sequence"}:
        raise ValueError("sample must be nonempty and distinct from annotation columns")
    library=_reference(reference)
    def anchor(value):
        if isinstance(value,bytes): return value.upper()
        if isinstance(value,str): return value.upper().encode('ascii')
        raise TypeError("Anchors must be strings or bytes")
    table,metrics=count_extended(library,fastq_path,sample,orientation,anchor(left_anchor),
        anchor(right_anchor),max_reads,policy,max_extra,anchor_mismatch)
    import pandas as pd
    frame=pd.DataFrame({"sgRNA":table.ids,"Gene":table.genes,
        "Sequence":[s.decode('ascii') for s in library.sequences],sample:table.counts[:,0]})
    frame.attrs['readsparrow']={"sample_columns":[sample],"total_reads":{sample:metrics['total_reads']}}
    return CountResult(frame,{**summarize_qc(frame),**metrics},dict(metrics['categories']))

def summarize_qc(counts_df,*,count_columns=None,total_reads=None):
    """Descriptive QC; flat dict for one sample, sample-keyed dict for many.

    Mapping rate is None without input totals. Explicit total_reads overrides
    DataFrame attrs. Guide representation does not estimate cellular coverage.
    """
    import pandas as pd
    if not isinstance(counts_df,pd.DataFrame) or not len(counts_df) or not counts_df.columns.is_unique:
        raise ValueError("Expected a nonempty DataFrame with unique columns")
    meta=counts_df.attrs.get('readsparrow',{})
    cols=list(count_columns) if count_columns is not None else meta.get('sample_columns')
    if cols is None:
        annotations={"sgRNA","ID","guide_id","Gene","gene_symbol","gene","Sequence","seq","sequence","contains_GTTTT"}
        cols=[c for c in counts_df if c not in annotations]
    if not cols or len(cols)!=len(set(cols)) or any(c not in counts_df for c in cols):
        raise ValueError("Select unique existing count columns")
    totals=total_reads if total_reads is not None else meta.get('total_reads',{})
    if not isinstance(totals,dict):
        if len(cols)!=1: raise ValueError("Supply total_reads per sample for multiple samples")
        totals={cols[0]:totals}
    result={}
    for col in cols:
        series=counts_df[col]
        if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            raise ValueError(f"Counts in {col!r} must be numeric integers")
        values=series.to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values<0) or np.any(values!=np.floor(values)):
            raise ValueError(f"Counts in {col!r} must be finite nonnegative integers")
        assigned=sum(int(v) for v in series)
        total=totals.get(col)
        if total is not None:
            if isinstance(total,(bool,np.bool_)) or not isinstance(total,(int,np.integer)) or total<assigned or total<0:
                raise ValueError("total_reads must be an integer >= assigned reads")
            total=int(total)
        result[col]={"sample":col,"guides":len(values),"assigned_reads":assigned,"total_reads":total,
            "mapping_rate_percent":100*assigned/total if total else None,
            "representation_percent":100*float((values>0).mean()),"zero_guides":int((values==0).sum()),
            "mean_count":float(values.mean()),"median_count":float(np.median(values)),
            "raw_count_gini":gini(values),"log1p_count_gini":gini(np.log1p(values))}
    return next(iter(result.values())) if len(cols)==1 else result
