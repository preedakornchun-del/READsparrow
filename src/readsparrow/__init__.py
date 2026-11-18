"""Streaming counting/QC. Legacy statistical callers remain experimental."""
__version__ = "0.12.1"
__all__ = ["count", "summarize_qc", "CountResult", "plots"]

def count(fastq_path, reference, orientation="auto", anchor_mismatch=0, **kwargs):
    """Count one FASTQ; see readsparrow.api.count for all options."""
    from .api import count as implementation
    return implementation(fastq_path, reference, orientation, anchor_mismatch, **kwargs)

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
    if name == "CountResult":
        from .api import CountResult
        return CountResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
