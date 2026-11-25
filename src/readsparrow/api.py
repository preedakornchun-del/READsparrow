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


@dataclass
class FlexibleCountResult:
    """In-memory result for the supplied-adapter ``count-flex`` workflow.

    ``counts_df`` contains only uniquely identifiable guide IDs. The complete
    reference remains available in ``guide_audit_df`` and duplicate spacer
    sequences are measured once in ``sequence_counts_df``. This prevents an
    observed sequence-group count from being silently copied to multiple IDs.
    """

    counts_df: object
    sequence_counts_df: object
    guide_audit_df: object
    gene_counts_df: object
    qc_summary: dict
    rejection_stats: dict

    @property
    def resolved_counts_df(self):
        """Alias documenting that ``counts_df`` contains identifiable IDs."""
        return self.counts_df

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


def _group_reference(reference):
    from .flex_counting import GroupLibrary, load_group_library
    if isinstance(reference, GroupLibrary):
        return reference
    if isinstance(reference, (str, Path)):
        return load_group_library(reference)
    import pandas as pd
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("reference must be a path, DataFrame or readsparrow.flex_counting.GroupLibrary")
    if not reference.columns.is_unique:
        raise ValueError("Reference column names must be unique")
    choices = [("sgRNA", "ID", "guide_id", "UID"),
               ("Gene", "gene_symbol", "gene", "gene_id"),
               ("Sequence", "seq", "sequence")]
    cols = [next((c for c in names if c in reference), None) for names in choices]
    if any(c is None for c in cols) or reference[cols].isna().any().any():
        raise ValueError("Reference needs nonmissing ID, gene and sequence columns")
    try:
        return GroupLibrary(reference[cols[0]].astype(str).tolist(),
                            reference[cols[1]].astype(str).tolist(),
                            [str(s).upper().encode("ascii") for s in reference[cols[2]]])
    except UnicodeEncodeError as exc:
        raise ValueError("Reference sequences must be ASCII DNA") from exc


def _flexible_frames(library, counts, sample):
    """Build the same four logical tables emitted by the count-flex CLI."""
    import pandas as pd
    from collections import Counter

    sequence_rows = []
    gene_counts = Counter()
    for group_id, sequence, members, value in zip(
            library.group_ids, library.group_sequences, library.members, counts):
        genes = sorted({library.genes[i] for i in members})
        count_value = int(value)
        if len(genes) == 1:
            gene_counts[genes[0]] += count_value
        sequence_rows.append({
            "sequence_group": group_id,
            "Sequence": sequence.decode("ascii"),
            "member_ids": ";".join(sorted(library.ids[i] for i in members)),
            "genes": ";".join(genes),
            "n_ids": len(members),
            "gene_identifiable": len(genes) == 1,
            "count": count_value,
        })

    guide_rows = []
    resolved_rows = []
    for index, sequence in enumerate(library.sequences):
        group = library.lookup[sequence]
        identifiable = len(library.members[group]) == 1
        value = int(counts[group])
        guide_rows.append({
            "sgRNA": library.ids[index],
            "Gene": library.genes[index],
            "Sequence": sequence.decode("ascii"),
            "sequence_group": library.group_ids[group],
            "id_identifiable": identifiable,
            "resolved_count": value if identifiable else pd.NA,
            "compatible_group_reads": value,
        })
        if identifiable:
            resolved_rows.append({"sgRNA": library.ids[index],
                                  "Gene": library.genes[index], sample: value})

    sequence_frame = pd.DataFrame(sequence_rows)
    guide_frame = pd.DataFrame(guide_rows)
    guide_frame["resolved_count"] = guide_frame["resolved_count"].astype("Int64")
    resolved_frame = pd.DataFrame(resolved_rows, columns=["sgRNA", "Gene", sample])
    gene_frame = pd.DataFrame(
        [{"gene": gene, "count": value} for gene, value in sorted(gene_counts.items())],
        columns=["gene", "count"],
    )
    return resolved_frame, sequence_frame, guide_frame, gene_frame

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


def count_guides(fastq_path, reference, *, mode="count-flex", sample="count",
                 orientation=None, adapter=None, max_adapter_edits=2,
                 max_reads=None, **kwargs):
    """Count one FASTQ through a stable Python API.

    Parameters
    ----------
    mode : {"count-flex", "count"}
        ``count-flex`` uses a supplied full upstream adapter, permits bounded
        adapter edits and always requires an exact reference spacer. ``count``
        dispatches to the anchor-bounded :func:`count` API.

    Notes
    -----
    ``count-flex`` requires an explicit forward/reverse orientation and an
    adapter. Duplicate spacer sequences are retained as sequence groups; their
    counts are never allocated to multiple guide IDs.
    """
    if mode == "count":
        if adapter is not None or max_adapter_edits != 2:
            raise ValueError("adapter and max_adapter_edits apply only to mode='count-flex'")
        selected_orientation = "auto" if orientation is None else orientation
        return count(fastq_path, reference, orientation=selected_orientation,
                     sample=sample, max_reads=max_reads, **kwargs)
    if mode != "count-flex":
        raise ValueError("mode must be 'count-flex' or 'count'")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported count-flex option(s): {names}")
    if orientation not in {"forward", "reverse"}:
        raise ValueError("count-flex requires orientation='forward' or 'reverse'")
    if adapter is None:
        raise ValueError("count-flex requires a full upstream adapter")
    if (isinstance(max_adapter_edits, (bool, np.bool_)) or
            not isinstance(max_adapter_edits, (int, np.integer)) or
            max_adapter_edits not in {0, 1, 2}):
        raise ValueError("max_adapter_edits must be integer 0, 1 or 2")
    if not isinstance(sample, str) or not sample or sample in {"sgRNA", "Gene", "Sequence"}:
        raise ValueError("sample must be nonempty and distinct from annotation columns")
    if isinstance(adapter, str):
        try:
            adapter_bytes = adapter.upper().encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("adapter must be ASCII A/C/G/T/N") from exc
    elif isinstance(adapter, bytes):
        adapter_bytes = adapter.upper()
    else:
        raise TypeError("adapter must be a string or bytes")

    from .flex_counting import count_flexible
    library = _group_reference(reference)
    values, metrics = count_flexible(library, fastq_path, sample, adapter_bytes,
                                     int(max_adapter_edits), orientation, max_reads)
    resolved, sequences, audit, genes = _flexible_frames(library, values, sample)
    attrs = {"mode": "count-flex", "sample_columns": [sample],
             "total_reads": {sample: metrics["total_reads"]}}
    for frame in (resolved, sequences, audit, genes):
        frame.attrs["readsparrow"] = attrs
    return FlexibleCountResult(resolved, sequences, audit, genes, metrics,
                               dict(metrics["categories"]))

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
