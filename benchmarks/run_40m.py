"""One predeclared40M-record engineering memory check; not independent samples."""
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent
import argparse
p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
from generate_input import generate
input_dir=out/'input'; generate(input_dir)
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':40000000,
    'reference_guides':1000,'input':'80 sequential copies of the fixed500000-record software fixture; repeated read IDs allowed; not independent biological replicates.',
    'purpose':'Check actual per-process RSS below8GiB while streaming40Mrecords. Not a new speed comparison or biological accuracy validation.',
    'runs':1,'timeout_seconds':900,'expected_assigned':32000000,'primary27timers_already_complete':'not assumed for this independent reproduction',
    'source_identity':'installed/source package; record version below','compressed_input_creation_memory':'1MiB transfer buffer',
    'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'protocol.json').open('x').write(json.dumps(protocol,indent=2))
fastq=out/'repeated_40m.fastq.gz'
with gzip.open(fastq,'wb',compresslevel=1) as target:
    for _ in range(80):
        with (input_dir/'sample.fastq').open('rb') as source:
            shutil.copyfileobj(source,target,length=1024*1024)
env=os.environ.copy(); env['PYTHONPATH']=str(BASE.parent/'src')
env.update({k:'1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE']})
before=time.monotonic()
with (out/'count.log').open('w') as log:
    subprocess.run([sys.executable,'-B',str(BASE/'benchmark_worker.py'),'api',str(fastq),
                    str(input_dir/'reference.tsv'),'40000000',str(out/'result.json')],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
result=json.loads((out/'result.json').read_text())
assert result['metrics']['total_reads']==40000000
assert sum(result['counts'])==32000000
assert result['metrics']['categories']=={'assigned_exact':32000000,'no_left_anchor':4000000,
                                        'anchor_region_no_exact_reference_match':4000000}
assert result['peak_RSS_bytes']<8*1024**3
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'records':40000000,
    'reference_guides':1000,'assigned_reads':32000000,'peak_RSS_MiB':result['peak_RSS_bytes']/1024**2,
    'CPU_seconds':result['CPU_seconds'],'elapsed_seconds':time.monotonic()-before,
    'input_gzip_bytes':fastq.stat().st_size,'input_sha256':hashlib.file_digest(fastq.open('rb'),'sha256').hexdigest(),
    'reference_sha256':hashlib.sha256((input_dir/'reference.tsv').read_bytes()).hexdigest(),
    'not_independent_biological_evidence':True,'not_a_full_genome_reference_memory_test':True}
(out/'verification.json').open('x').write(json.dumps(receipt,indent=2)); print(json.dumps(receipt),flush=True)
