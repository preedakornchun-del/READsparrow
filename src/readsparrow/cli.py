"""Small standalone command line interface. No MAGeCK subprocesses or imports."""
from __future__ import annotations

import argparse
import html
import json
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from . import __version__
from .counting import count_fastq, load_library
from .orientation import count_extended
from .io import CountTable, load_counts, new_output_dir, read_rows, save_counts, sha256, write_json, write_tsv
from .qc import summarize
from .simulation import SimulationConfig, write_simulation


def _load_legacy_inference():
    global scipy, calibration_suite, comparison_suite, TestConfig, call_hits, load_sample_sheet, make_design, GuideConfig, call_guide_hits, EmpiricalConfig, call_empirical_hits, null_comparison_suite, PowerConfig, call_power_hits, CorrelatedConfig, call_correlated_hits, power_comparison_suite, correlated_comparison_suite, ShapeConfig, call_shape_hits, MeasurementConfig, call_measurement_hits, JointConfig, call_joint_hits, StableConfig, call_stable_hits
    try:
        import scipy
        from .calibration import calibration_suite
        from .comparison import comparison_suite
        from .inference import TestConfig, call_hits, load_sample_sheet, make_design
        from .guide_inference import GuideConfig, call_guide_hits
        from .empirical_inference import EmpiricalConfig, call_empirical_hits
        from .null_comparison import comparison_suite as null_comparison_suite
        from .power_inference import PowerConfig, call_power_hits
        from .correlated_inference import CorrelatedConfig, call_correlated_hits
        from .power_comparison import comparison_suite as power_comparison_suite
        from .correlated_comparison import run_suite as correlated_comparison_suite
        from .shape_inference import ShapeConfig, call_shape_hits
        from .measurement_inference import MeasurementConfig, call_measurement_hits
        from .joint_inference import JointConfig, call_joint_hits
        from .stable_inference import StableConfig, call_stable_hits
    except ImportError as exc:
        if exc.name and exc.name.startswith('scipy'):
            raise ValueError('Legacy hit calling requires pip install "readsparrow[legacy]"') from exc
        raise

def _scipy_version():
    from importlib.metadata import version, PackageNotFoundError
    try:
        return version('scipy')
    except PackageNotFoundError:
        return None


def provenance(inputs, started):
    return {"readsparrow_version": __version__, "python": sys.version, "numpy": np.__version__, "scipy": _scipy_version(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "argv": sys.argv, "wall_seconds_before_manifest_hashing": time.perf_counter() - started,
        "inputs": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in dict.fromkeys(map(str, inputs))],
        "package_sources": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "MAGeCK_required": False, "experimental_hit_caller": True}


def write_analysis(out, table, metadata, args):
    controls = []
    if args.control_guides:
        controls = [s.strip() for s in args.control_guides.read_text().splitlines() if s.strip()]
        if len(controls) != len(set(controls)):
            raise ValueError("Duplicate control guide IDs")
    settings = dict(normalization=args.normalization, min_count=args.min_count, min_guides=args.min_guides, fdr=args.fdr)
    if args.method == "baseline":
        results, info = call_hits(table, metadata, args.treatment, args.control, TestConfig(**settings), controls)
    else:
        settings.update(minimum_support=args.minimum_support, support_fraction=args.support_fraction,
                        min_guide_count=args.min_guide_count, adjustment=args.adjustment)
        if args.method in {"corr-adaptive", "shape-adaptive", "measurement-adaptive", "joint-adaptive", "stable-adaptive"}:
            blocks=None
            if args.control_blocks:
                headers,block_rows=read_rows(args.control_blocks)
                if not {"sgRNA","block"}.issubset(headers):
                    raise ValueError("Control block file requires sgRNA and block columns")
                blocks={r["sgRNA"]:r["block"] for r in block_rows}
                if len(blocks)!=len(block_rows):
                    raise ValueError("Duplicate guide ID in control blocks")
            options=dict(**settings, min_null_controls=args.min_null_controls,
                correlation_mode="matched-blocks" if blocks is not None else "envelope",
                correlation_upper=args.correlation_upper,coherent_weight=args.coherent_weight)
            if args.method == "stable-adaptive":
                config=StableConfig(**options)
                results,info,guides=call_stable_hits(table,metadata,args.treatment,args.control,config,controls,blocks)
            elif args.method == "joint-adaptive":
                config=JointConfig(**options)
                results,info,guides=call_joint_hits(table,metadata,args.treatment,args.control,config,controls,blocks)
            elif args.method == "measurement-adaptive":
                config=MeasurementConfig(**options)
                results,info,guides=call_measurement_hits(table,metadata,args.treatment,args.control,config,controls,blocks)
            elif args.method == "shape-adaptive":
                config=ShapeConfig(**options, matched_inference=args.matched_inference)
                results,info,guides=call_shape_hits(table,metadata,args.treatment,args.control,config,controls,blocks)
            else:
                config=CorrelatedConfig(**options)
                results,info,guides=call_correlated_hits(table,metadata,args.treatment,args.control,config,controls,blocks)
            write_tsv(out/"gene_subset_audit.tsv",list(info["subset_audit"][0]),info["subset_audit"])
            write_tsv(out/"null_control_audit.tsv",list(info["control_audit"][0]),info["control_audit"])
        elif args.method in {"power-bounded", "power-loo"}:
            config = PowerConfig(**settings, aggregation="bounded" if args.method == "power-bounded" else "loo",
                min_null_controls=args.min_null_controls, background_correlation=args.guide_background_correlation,
                clipping_threshold=args.clipping_threshold)
            results, info, guides = call_power_hits(table, metadata, args.treatment, args.control, config, controls)
            write_tsv(out / "gene_subset_audit.tsv", list(info["subset_audit"][0]), info["subset_audit"])
            write_tsv(out / "null_control_audit.tsv", list(info["control_audit"][0]), info["control_audit"])
        elif args.method == "guide-aware":
            results, info, guides = call_guide_hits(table, metadata, args.treatment, args.control, GuideConfig(**settings), controls)
        else:
            config = EmpiricalConfig(**settings, null_model=args.method, min_null_controls=args.min_null_controls)
            results, info, guides = call_empirical_hits(table, metadata, args.treatment, args.control, config, controls)
            write_tsv(out / "null_control_audit.tsv", list(info["control_audit"][0]), info["control_audit"])
        write_tsv(out / "guide_results.tsv", list(guides[0]), guides)
    write_tsv(out / "gene_results.tsv", list(results[0]), results)
    write_json(out / "analysis.json", info)
    qc = summarize(table)
    write_tsv(out / "sample_qc.tsv", list(qc[0]), qc)
    tested = sorted((r for r in results if r["tested"]), key=lambda r: (r["qvalue"], r["gene"]))[:50]
    columns = ["gene", "guides", "log2_effect", "ci_low", "ci_high", "pvalue", "qvalue", "hit", "guide_direction_agreement", "max_guide_read_share", "diagnostic_flags"]
    cells = lambda r: "".join(f"<td>{html.escape(format(r[c], '.5g') if isinstance(r[c], float) else str(r[c]))}</td>" for c in columns)
    body = "".join(f"<tr>{cells(r)}</tr>" for r in tested)
    warnings = "".join(f"<li>{html.escape(w)}</li>" for w in info["warnings"])
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>READsparrow experimental analysis</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border:1px solid #ddd}}th{{background:#eaf2f8}}</style>
<h1>READsparrow standalone analysis</h1><p>Experimental alpha — not production-validated.</p>
<p>{html.escape(info['contrast'])}; {info['tested_genes']} tested genes. {html.escape(info['fdr_family'])}.</p>
<p>Method: {html.escape(info['method'])}. Median-guide, equal-guide-mean and count-weighted gene-sum effects are different estimands.</p>
<h2>Limitations and warnings</h2><ul>{warnings}</ul><h2>Top 50 by declared decision metric</h2>
<p>Guide diagnostics are descriptive, not robustness probabilities. Full tested and filtered gene lists are in gene_results.tsv.</p>
<table><tr>{''.join('<th>'+html.escape(c)+'</th>' for c in columns)}</tr>{body}</table></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")
    return info


def parser():
    p = argparse.ArgumentParser(description="READsparrow standalone alpha: counting, QC, experimental hit calling")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    count = sub.add_parser("count", help="Count one FASTQ with legacy-compatible exact extraction")
    count.add_argument("--library", type=Path, required=True)
    count.add_argument("--fastq", type=Path, required=True)
    count.add_argument("--sample", required=True)
    count.add_argument("--orientation", choices=["forward", "reverse", "auto", "mixed"], required=True)
    count.add_argument("--anchor-mismatch", type=int, choices=[0, 1], default=0)
    count.add_argument("--left-anchor", default="CACCG")
    count.add_argument("--right-anchor", default="GTTTT")
    count.add_argument("--max-reads", type=int)
    count.add_argument("--policy", choices=["legacy", "unique"], default="legacy")
    count.add_argument("--max-extra", type=int, default=32)
    count.add_argument("--output", type=Path, required=True)
    flex = sub.add_parser("count-flex", help="Full-reference sequence groups; adapter-led short-read counting")
    flex.add_argument("--library", type=Path, required=True)
    flex.add_argument("--fastq", type=Path, required=True)
    flex.add_argument("--sample", required=True)
    flex.add_argument("--orientation", choices=["forward", "reverse"], required=True)
    flex.add_argument("--adapter", required=True, help="Full upstream adapter in forward orientation, A/C/G/T/N")
    flex.add_argument("--max-adapter-edits", type=int, default=2)
    flex.add_argument("--max-reads", type=int)
    flex.add_argument("--output", type=Path, required=True)
    variant = sub.add_parser("count-variants", help="Exact counts plus separate observed variants and audit-only candidates")
    variant.add_argument("--library", type=Path, required=True)
    variant.add_argument("--fastq", type=Path, required=True)
    variant.add_argument("--sample", required=True)
    variant.add_argument("--orientation", choices=["forward", "reverse"], required=True)
    variant.add_argument("--adapter", required=True)
    variant.add_argument("--max-adapter-edits", type=int, default=2)
    variant.add_argument("--max-reads", type=int)
    variant.add_argument("--output", type=Path, required=True)
    qc = sub.add_parser("qc", help="Describe an integer count matrix")
    qc.add_argument("--counts", type=Path, required=True)
    qc.add_argument("--output", type=Path, required=True)
    for name in ("test", "run"):
        t = sub.add_parser(name, help="Standalone experimental gene test" if name == "test" else "FASTQs -> counts -> QC -> standalone test")
        t.add_argument("--counts" if name == "test" else "--library", type=Path, required=True)
        t.add_argument("--samples", type=Path, required=True)
        t.add_argument("--treatment", required=True)
        t.add_argument("--control", required=True)
        t.add_argument("--normalization", choices=["median", "total", "control", "none"], default="median")
        t.add_argument("--control-guides", type=Path)
        t.add_argument("--min-count", type=int, default=10)
        t.add_argument("--min-guides", type=int, default=2)
        t.add_argument("--fdr", type=float, default=.05)
        t.add_argument("--method", choices=["stable-adaptive", "joint-adaptive", "measurement-adaptive", "shape-adaptive", "corr-adaptive", "power-loo", "power-bounded", "background", "empirical", "guide-aware", "baseline"], default="background")
        t.add_argument("--matched-inference", choices=["posterior", "scan"], default="posterior", help="Shape method only: posterior expected FDP is model-dependent, NOT BH q-values")
        t.add_argument("--control-blocks",type=Path,help="Genuine matched null control groups: sgRNA,block TSV; NEVER randomly group NTCs")
        t.add_argument("--correlation-upper",type=float,default=1.,help="Prespecified nonnegative correlation bound when no matched blocks exist; default 1, not independence")
        t.add_argument("--coherent-weight",type=float,default=.8,help="Prespecified coherent/sparse union weight; do not tune on target hits")
        t.add_argument("--guide-background-correlation", type=float, default=0.)
        t.add_argument("--clipping-threshold", type=float, default=2.5)
        t.add_argument("--min-null-controls", type=int, default=100)
        t.add_argument("--min-guide-count", type=int, default=5)
        t.add_argument("--minimum-support", type=int, default=2)
        t.add_argument("--support-fraction", type=float, default=.5)
        t.add_argument("--adjustment", choices=["bh", "by"], default="bh")
        t.add_argument("--experimental", action="store_true", help="Required acknowledgement that the caller is research alpha")
        t.add_argument("--output", type=Path, required=True)
    sim = sub.add_parser("simulate", help="Write seeded count-level screens with biological gene truth")
    sim.add_argument("--output", type=Path, required=True)
    defaults = SimulationConfig()
    for f in fields(defaults):
        name = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            sim.add_argument(name, action="store_true", default=default)
        else:
            sim.add_argument(name, type=type(default), default=default)
    cal = sub.add_parser("calibrate", help="Run frozen Monte Carlo diagnostic scenarios; expose failures")
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--repeats", type=int, default=50)
    cal.add_argument("--genes", type=int, default=1000)
    cal.add_argument("--seed", type=int, default=826261000)
    cal.add_argument("--fail-on-baseline-flags", action="store_true")
    cal.add_argument("--baseline-only", action="store_true", help="Reproduce the earlier baseline-only calibration workflow")
    cal.add_argument("--guide-only", action="store_true", help="Reproduce the v0.3 two-engine suite instead of the v0.4 four-engine suite")
    cal.add_argument("--null-repeats", type=int, default=100)
    cal.add_argument("--previous-null-only", action="store_true", help="Reproduce v0.4 four-engine calibration instead of power comparison")
    cal.add_argument("--previous-power-only",action="store_true",help="Reproduce v0.5 power comparison rather than covariance/sparse suite")
    cal.add_argument("--shape-suite", action="store_true", help="Run v0.7 paired shape/sparse evaluation; preserve old default suite")
    cal.add_argument("--scenarios", nargs="+", help="Optional named cases from the guide-aware paired calibration suite")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    started = time.perf_counter()
    inputs = []
    try:
        if args.command in {"test", "run"} and not args.experimental:
            raise ValueError("Hit calling is experimental. Read docs/METHODS.md and explicitly pass --experimental")
        if args.command in {"test", "run", "calibrate"}:
            _load_legacy_inference()
        out = new_output_dir(args.output)
        exit_code = 0
        if args.command == "count-variants":
            from .flex_counting import load_group_library
            from .variant_counting import count_variants, save_variants
            lib = load_group_library(args.library)
            values, metrics, variants, candidates = count_variants(lib, args.fastq, args.sample,
                args.adapter.upper().encode('ascii'), out/'candidate_read_annotations.tsv.gz',
                args.max_adapter_edits, args.orientation, args.max_reads)
            save_variants(out, lib, values, metrics, variants, candidates, args.sample)
            inputs = [args.library, args.fastq]
        elif args.command == "count-flex":
            from .flex_counting import load_group_library, count_flexible, save_flexible
            lib = load_group_library(args.library)
            values, metrics = count_flexible(lib, args.fastq, args.sample,
                args.adapter.upper().encode('ascii'), args.max_adapter_edits, args.orientation, args.max_reads)
            save_flexible(out, lib, values, metrics, args.sample)
            inputs = [args.library, args.fastq]
        elif args.command == "count":
            lib = load_library(args.library)
            table, metrics = count_extended(lib, args.fastq, args.sample, args.orientation,
                                         args.left_anchor.upper().encode("ascii"), args.right_anchor.upper().encode("ascii"), args.max_reads,
                                         policy=args.policy, max_extra=args.max_extra, anchor_mismatch=args.anchor_mismatch)
            save_counts(out / "counts.tsv", table)
            write_json(out / "count_metrics.json", metrics)
            inputs = [args.library, args.fastq]
        elif args.command == "qc":
            table = load_counts(args.counts)
            rows = summarize(table)
            write_tsv(out / "sample_qc.tsv", list(rows[0]), rows)
            inputs = [args.counts]
        elif args.command in {"test", "run"}:
            metadata = load_sample_sheet(args.samples)
            # Explicitly subset by condition; reject selected duplicated samples in the design validator.
            metadata = [r for r in metadata if r["condition"] in {args.treatment, args.control}]
            selected = [r["sample"] for r in metadata]
            make_design(selected, metadata, args.treatment, args.control)
            inputs = [args.samples]
            if args.command == "test":
                table = load_counts(args.counts, selected)
                inputs.append(args.counts)
            else:
                lib = load_library(args.library)
                inputs.append(args.library)
                columns, metrics = [], []
                for row in metadata:
                    if not row.get("fastq") or row.get("orientation") not in {"forward", "reverse"}:
                        raise ValueError("run sample sheet requires fastq and explicit orientation for each sample")
                    path = Path(row["fastq"])
                    if not path.is_absolute():
                        path = args.samples.resolve().parent / path
                    one, qc = count_fastq(lib, path, row["sample"], row["orientation"],
                                         row.get("left_anchor", "CACCG").upper().encode(), row.get("right_anchor", "GTTTT").upper().encode(),
                                         policy=row.get("policy", "legacy"), max_extra=int(row.get("max_extra", "32")))
                    columns.append(one.counts[:, 0])
                    metrics.append({"sample": row["sample"], **qc})
                    inputs.append(path)
                table = CountTable(lib.ids, lib.genes, selected, np.column_stack(columns))
                save_counts(out / "counts.tsv", table)
                write_json(out / "count_metrics.json", metrics)
            if args.control_guides:
                inputs.append(args.control_guides)
            if args.control_blocks:
                inputs.append(args.control_blocks)
            info = write_analysis(out, table, metadata, args)
            print(f"Experimental standalone test: {info['tested_genes']} target genes; no MAGeCK required")
        elif args.command == "simulate":
            write_simulation(out, SimulationConfig(**{f.name: getattr(args, f.name) for f in fields(SimulationConfig())}))
        else:
            if args.baseline_only:
                report = calibration_suite(out, args.repeats, args.genes, args.seed,
                                           lambda s: print(json.dumps(s), flush=True))
                if args.fail_on_baseline_flags and not report["baseline_diagnostic_pass"]:
                    exit_code = 3
            else:
                if args.fail_on_baseline_flags:
                    raise ValueError("--fail-on-baseline-flags applies only with --baseline-only")
                if args.shape_suite:
                    from .shape_comparison import run_suite
                    run_suite(out, args.repeats, args.null_repeats, args.genes, args.seed, args.scenarios,
                              lambda s: print(json.dumps(s), flush=True))
                elif args.guide_only:
                    comparison_suite(out, args.repeats, args.genes, args.seed, args.scenarios,
                                     lambda s: print(json.dumps(s), flush=True))
                elif args.previous_null_only:
                    null_comparison_suite(out, args.repeats, args.genes, args.seed, args.scenarios,
                                          lambda s: print(json.dumps(s), flush=True), args.null_repeats)
                elif args.previous_power_only:
                    power_comparison_suite(out, args.repeats, args.null_repeats, args.genes, args.seed, args.scenarios,
                                           lambda s: print(json.dumps(s), flush=True))
                else:
                    correlated_comparison_suite(out,args.repeats,args.null_repeats,args.genes,args.seed,args.scenarios,
                        lambda s:print(json.dumps(s),flush=True))
        write_json(out / "provenance.json", provenance(inputs, started))
        return exit_code
    except (ValueError, OSError, UnicodeError) as exc:
        p.exit(2, f"READsparrow: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
