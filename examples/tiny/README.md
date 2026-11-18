# Small synthetic example

Six fixtures,100–112 reads each, four20nt guides with one zero-count guide.
Expected counts are specified by construction in `scripts/generate_demo.py`,
which imports neither READsparrow nor NumPy/Pandas. Expected Ginis are computed
independently using the sorted-sum definition. These are engineering fixtures,
not biological replicates, neutral controls or evidence for biological FDR.

| Case | Expected guide counts | Assigned / input |
|---|---|---|
| strict forward / reverse |40,25,15,0|80/100|
| auto forward / reverse |40,25,15,0|80/100|
| mixed |40,25,15,0|80/100|
| tolerant anchors |43,29,20,0|92/112|

Each case has10missing-anchor reads and10reads with an N-containing spacer.
The tolerant case adds3upstream,4downstream and5two-anchor substitutions.
The guide sequence itself must always match exactly.

Run `python scripts/run_demo.py --output demo-output` from the repository root.
The script verifies count tables exactly, QC/rejection JSON at a specified
numeric tolerance and all six PNG dimensions. The `expected` directory contains
reference images; font/rendering differences can change PNG bytes across systems.
The script refuses to overwrite an existing output directory.
