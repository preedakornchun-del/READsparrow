"""Regenerate the exact archived500k engineering fixture (seed2026111701)."""
import csv
import hashlib
import json
import random
from pathlib import Path

BASE=Path(__file__).resolve().parent
def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((BASE/'input_manifest.json').read_text())
    if not (output/'sample.fastq').exists():
        rng=random.Random(2026111701); sequences=[]
        while len(sequences)<1000:
            s=''.join(rng.choice('ACGT') for _ in range(20))
            if s not in sequences:sequences.append(s)
        with (output/'reference.tsv').open('w',newline='')as f:
            w=csv.writer(f,delimiter='\t');w.writerow(['sgRNA','Gene','Sequence'])
            w.writerows((f'g{i}',f'GENE{i}',s)for i,s in enumerate(sequences))
        with (output/'sample.fastq').open('wb')as f:
            for i in range(500000):
                s=('ACACACAC'+'CACCG'+sequences[i%1000]+'GTTTT').encode()
                if i%10==8:s=b'N'*40
                if i%10==9:s=b'CACCG'+b'N'*20+b'GTTTT'
                f.write(b'@r%d\n'%i+s+b'\n+\n'+b'I'*len(s)+b'\n')
    for n,h in manifest.items():
        with (output/n).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,n
    return output

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(generate(p.parse_args().output))
