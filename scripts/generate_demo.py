"""Build small deterministic synthetic inputs and independent expected results.

No readsparrow/numpy/pandas imports: expected counts come from construction.
This is an engineering fixture, never biological ground truth or replication.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/tiny'
GUIDES = ['ACGTGCACTGATCGATGACA', 'AAAGTTTTCCCCCCCCCCCC',
          'AAACACCGCCCCCCCCCCCC', 'TGCATGCATGCATGCATGCA']

def rev(s):
    return s.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]

def gini(values):
    x = sorted(values)
    return sum((2*i-len(x)-1)*v for i,v in enumerate(x, 1))/(len(x)*sum(x)) if sum(x) else 0.

def main():
    assert all(len(g) == 20 for g in GUIDES)
    (BASE / 'expected').mkdir(parents=True, exist_ok=True)
    with (BASE / 'reference.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['sgRNA', 'Gene', 'Sequence'])
        w.writerows((f'g{i+1}', f'GENE{i+1}', g) for i,g in enumerate(GUIDES))
    valid = ['CACCG'+g+'GTTTT' for g,n in zip(GUIDES, [40,25,15,0]) for _ in range(n)]
    invalid = ['N'*30]*10 + ['CACCG'+'N'*20+'GTTTT']*10
    cases = {
        'strict_forward': (valid+invalid, 'forward', 0, [40,25,15,0]),
        'strict_reverse': ([rev(s) for s in valid+invalid], 'reverse', 0, [40,25,15,0]),
        'auto_forward': (valid+invalid, 'auto', 0, [40,25,15,0]),
        'auto_reverse': ([rev(s) for s in valid+invalid], 'auto', 0, [40,25,15,0]),
        'mixed': ([rev(s) if i%2 else s for i,s in enumerate(valid)]+invalid, 'mixed', 0, [40,25,15,0]),
        'tolerant': (valid+invalid+['CATCG'+GUIDES[0]+'GTTTT']*3+
                     ['CACCG'+GUIDES[1]+'GTTTA']*4+['CATCG'+GUIDES[2]+'GTTTA']*5,
                     'forward', 1, [43,29,20,0])}
    manifest = {}
    for name,(reads,orientation,mismatch,counts) in cases.items():
        fastq = BASE / (name+'.fastq')
        with fastq.open('w') as f:
            for i,s in enumerate(reads):
                f.write(f'@{name}_{i}\n{s}\n+\n'+('I'*len(s))+'\n')
        expected = BASE / 'expected'
        with (expected/(name+'_counts.tsv')).open('w', newline='') as f:
            w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['sgRNA','Gene','Sequence','count'])
            w.writerows((f'g{i+1}',f'GENE{i+1}',g,n) for i,(g,n) in enumerate(zip(GUIDES,counts)))
        categories = {'assigned_exact':sum(counts), 'no_left_anchor':10,
                      'anchor_region_no_exact_reference_match':10}
        qc = {'sample':'count','guides':4,'assigned_reads':sum(counts),'total_reads':len(reads),
              'mapping_rate_percent':100*sum(counts)/len(reads), 'representation_percent':75.,
              'zero_guides':1,'mean_count':sum(counts)/4,'median_count':(counts[1]+counts[2])/2,
              'raw_count_gini':gini(counts),'log1p_count_gini':gini([math.log1p(v) for v in counts])}
        qc={k:round(v,12) if isinstance(v,float) else v for k,v in qc.items()}
        (expected/(name+'_qc.json')).write_text(json.dumps(qc,indent=2,sort_keys=True)+'\n')
        (expected/(name+'_rejections.json')).write_text(json.dumps(categories,indent=2,sort_keys=True)+'\n')
        manifest[name]={'fastq':fastq.name,'orientation':orientation,'anchor_mismatch':mismatch,
                        'records':len(reads),'expected_counts':counts,'sha256':hashlib.sha256(fastq.read_bytes()).hexdigest()}
    (BASE/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print('Generated six constructed fixtures; expected counts/QC independent of READsparrow.')

if __name__ == '__main__':
    main()
