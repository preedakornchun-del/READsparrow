"""Fixed sequential27-unit software benchmark. Resume only completed records."""
import csv
import datetime
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; PROJECT=ROOT.parent
protocol=json.loads((BASE/'benchmark_protocol.json').read_text())
import argparse
p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--resume',action='store_true'); args=p.parse_args()
out=args.output.resolve(); out.mkdir(parents=True,exist_ok=args.resume)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=out/'frozen.json'
if not freeze.exists():
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/readsparrow').glob('*.py')}
    record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':sha(BASE/'benchmark_protocol.json'),
            'sources':sources,'worker_sha256':sha(BASE/'benchmark_worker.py')}
    freeze.open('x').write(json.dumps(record,indent=2))
frozen=json.loads(freeze.read_text())
assert sha(BASE/'benchmark_protocol.json')==frozen['protocol_sha256']
assert sha(BASE/'benchmark_worker.py')==frozen['worker_sha256']
for name,h in frozen['sources'].items(): assert sha(ROOT/name)==h,name
from generate_input import generate
input_dir=out/'input'; generate(input_dir)
fastq=input_dir/'sample.fastq'; reference=input_dir/'reference.tsv'
if not (out/'input_manifest.json').exists():
    (out/'input_manifest.json').open('x').write((BASE/'input_manifest.json').read_text())
units=[]
modes=protocol['primary_modes']
for rep in range(3):
    for mode in modes[rep:]+modes[:rep]: units.append((f'primary_{rep}_{mode}',mode,100000))
for rep in range(3):
    for n in protocol['memory_prefix_sizes']: units.append((f'memory_{rep}_{n}','api',n))
start=time.monotonic()
for name,mode,n in units:
    target=out/(name+'.json')
    if target.exists(): continue
    assert time.monotonic()-start<protocol['total_wall_budget_seconds']
    env=os.environ.copy(); env.update({k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE']})
    env['PYTHONPATH']=str(BASE/'baseline_v012' if mode=='baseline' else ROOT/'src')
    before=time.monotonic()
    temporary=target.with_suffix('.tmp')
    command=[sys.executable,'-B',str(BASE/'benchmark_worker.py'),mode,str(fastq),str(reference),str(n),str(temporary)]
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    record=json.loads(temporary.read_text()); record['external_elapsed_seconds']=time.monotonic()-before
    record['unit']=name; temporary.write_text(json.dumps(record,indent=2)); temporary.replace(target)
    print(json.dumps({'unit':name,'CPU_seconds':record['CPU_seconds'],'RSS_MiB':record['peak_RSS_bytes']/1024**2}),flush=True)
records=[json.loads((out/(name+'.json')).read_text()) for name,_,_ in units]
baseline=next(r for r in records if r['mode']=='baseline')
strict=[r for r in records if r['mode']=='strict']
timing_keys={'loop_wall_seconds','loop_cpu_seconds'}
for r in strict:
    assert r['counts']==baseline['counts']
    assert {k:v for k,v in r['metrics'].items() if k not in timing_keys}=={k:v for k,v in baseline['metrics'].items() if k not in timing_keys}
rows=[]
for mode in modes:
    group=[r for r in records if r['unit'].startswith('primary') and r['mode']==mode]
    rows.append({'mode':mode,'repeats':len(group),**{metric:statistics.median(r[metric] for r in group) for metric in ['CPU_seconds','external_elapsed_seconds','peak_RSS_bytes']},
        'loop_CPU_seconds':statistics.median(r['metrics']['loop_cpu_seconds'] for r in group),
        'CPU_min':min(r['CPU_seconds'] for r in group),'CPU_max':max(r['CPU_seconds'] for r in group)})
memory=[{'reads':n,'RSS_MiB_median':statistics.median(r['peak_RSS_bytes']/1024**2 for r in records if r['unit'].startswith('memory') and r['records']==n)} for n in protocol['memory_prefix_sizes']]
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_units':len(records),'timing':rows,'memory_prefixes':memory,
         'strict_counts_rejections_non_timing_metrics_equal_baseline':True,'original_default_counting_source_byte_identical':sha(ROOT/'src/readsparrow/counting.py')==sha(BASE/'baseline_v012/readsparrow/counting.py'),
         'measured40Mreads':False,'sample_truth_is_engineering_fixture':True}
(out/'summary.json').open('w').write(json.dumps(summary,indent=2)); print(json.dumps(summary),flush=True)
