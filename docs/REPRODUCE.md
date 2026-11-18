# Exact reproduction commands

Use a NEW environment and empty output folders. CPython3.11 is the locked
reference environment; the public API supports Python>=3.10 with compatible
dependencies. The pinned lock is intended for3.11, not3.10.

```sh
git clone https://github.com/preedakornchun-del/READsparrow.git
cd READsparrow
git checkout v0.12.1
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation .
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
```

On Windows use `.venv\Scripts\activate` and the corresponding environment-variable
syntax. The resource harness uses Unix `resource` and is intended for Linux/macOS.

## Main counting/QC example and figures (normally under a minute)

```sh
python scripts/run_demo.py --output demo-output
python scripts/run_demo.py --output demo-output --verify-only
python -m pytest -q --junitxml=regression.xml
```

The six `*_counts.tsv`, `*_qc.json`, `*_rejections.json`, `*_qc.png` files and
`verification.json` are written into `demo-output`. Expected results are in
`examples/tiny/expected`; input identities and settings are in
`examples/tiny/manifest.json`. Numeric QC uses tolerance1e-11; PNG rendering may
vary by platform/fonts, so image identity is not a cross-platform correctness gate.

## CLI and API

```sh
readsparrow --version
readsparrow count --library examples/tiny/reference.tsv \
  --fastq examples/tiny/strict_forward.fastq --sample demo \
  --orientation forward --output cli-output
readsparrow qc --counts cli-output/counts.tsv --output cli-qc
```

```python
import readsparrow as rs
result = rs.count('examples/tiny/strict_forward.fastq',
                  'examples/tiny/reference.tsv', orientation='forward')
counts, qc, rejections = result
print(counts)
print(qc['mapping_rate_percent'])  #80.0
```

The counting step knows input totals. A stand-alone QC table cannot reconstruct
mapping rate from counts without total reads; supply `total_reads=` in the API.

## CPU/RAM engineering measurements

```sh
python benchmarks/generate_input.py --output benchmark-input
python benchmarks/run_resources.py --output benchmark-output
# Only if that same run was interrupted; existing completed records are retained:
python benchmarks/run_resources.py --output benchmark-output --resume
```

The generator verifies both original input SHA256 hashes. It creates about42MiB
of uncompressed input:500kreads/1000guides, seed2026111701. The harness runs
27fixed units sequentially: six modes x three100k-read repeats, then three
memory prefixes x three repeats. Baseline source is preserved under
`benchmarks/baseline_v012`. Results include individual records and `summary.json`.
The unchanged core is verified by SHA; timing values must be measured, not promised
to match historical values. Imports/DataFrame conversion count toward API timing.

Optional40M-record strict-forward API streaming memory check:

```sh
python benchmarks/run_40m.py --output benchmark-output-40m
```

This creates an approximately450MBgzip from80sequential copies of the500kfixture
and actually reads40Mrecords once. It checks32Massigned/4Mmissing-anchor/4Mno-match
and process RSS<8GiB. CPU/RAM depend on environment. Reference size remains1000;
this does not establish genome-wide-reference performance or independent
biological replication. `historical_*` JSON files are explicitly the previous
alpha measurements, not new measurements under the stable version. Historical
gzip headers may differ; deterministic identity is the decompressed payload.

## Build and install the archived wheel

```sh
python scripts/build_release.py
python -m pip install --no-deps dist/readsparrow-0.12.1-py3-none-any.whl
```

`dist/SHA256SUMS.txt` identifies the exact wheel/source archive produced by that
build. Tar/gzip build timestamps can differ; verify source/wheel contents and
their release-specific checksums. GitHub's release page and the Zenodo record
provide archived distribution files once publication is complete.
