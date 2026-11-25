"""Streaming counting/QC. Legacy statistical callers remain experimental."""
__version__ = "0.13.0"
__all__ = ["count", "count_guides", "summarize_qc", "CountResult",
           "FlexibleCountResult", "plots"]

def count(fastq_path, reference, orientation="auto", anchor_mismatch=0, **kwargs):
    """Count one FASTQ; see readsparrow.api.count for all options."""
    from .api import count as implementation
    return implementation(fastq_path, reference, orientation, anchor_mismatch, **kwargs)

def count_guides(fastq_path, reference, **kwargs):
    """Count through anchor-bounded or supplied-adapter mode; see API docs."""
    from .api import count_guides as implementation
    return implementation(fastq_path, reference, **kwargs)

def summarize_qc(counts_df, **kwargs):
    """Descriptive QC; mapping rates require total input read counts."""
    from .api import summarize_qc as implementation
    return implementation(counts_df, **kwargs)

def __getattr__(name):
    if name == "plots":
        import importlib
        module = importlib.import_module(".plots", __name__)
        globals()[name] = module
        return module
    if name in {"CountResult", "FlexibleCountResult"}:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
