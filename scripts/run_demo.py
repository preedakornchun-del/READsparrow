"""Run six tiny examples and verify independently specified outputs (<1min typically)."""
import argparse
import importlib.metadata
import json
import math
import platform
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/tiny'

def normalized(qc):
    keys = ['sample','guides','assigned_reads','total_reads','mapping_rate_percent',
            'representation_percent','zero_guides','mean_count','median_count',
            'raw_count_gini','log1p_count_gini']
    return {k:round(qc[k],12) if isinstance(qc[k],float) else qc[k] for k in keys}

def verify(output):
    import pandas as pd
    cases=json.loads((BASE/'manifest.json').read_text())
    for name in cases:
        pd.testing.assert_frame_equal(pd.read_csv(output/(name+'_counts.tsv'),sep='\t'),
                                     pd.read_csv(BASE/'expected'/(name+'_counts.tsv'),sep='\t'))
        for suffix in ['qc','rejections']:
            actual=json.loads((output/(name+'_'+suffix+'.json')).read_text())
            expected=json.loads((BASE/'expected'/(name+'_'+suffix+'.json')).read_text())
            assert actual.keys()==expected.keys()
            for k,v in expected.items():
                if isinstance(v,float): assert math.isclose(actual[k],v,rel_tol=1e-11,abs_tol=1e-11),(name,k,actual[k],v)
                else: assert actual[k]==v,(name,k,actual[k],v)
        png=(output/(name+'_qc.png')).read_bytes()
        assert png[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',png[16:24])==(1680,600)
    return {'passed':True,'cases':len(cases),'counts':'exact','QC':'numeric tolerance1e-11',
            'figures':'six PNGs at1680x600; visual appearance may differ by platform/fonts'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'demo-output');p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.output),indent=2));return
    args.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import readsparrow as rs
    cases=json.loads((BASE/'manifest.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    for name,case in cases.items():
        result=rs.count(BASE/case['fastq'],BASE/'reference.tsv',orientation=case['orientation'],anchor_mismatch=case['anchor_mismatch'])
        result.counts_df.to_csv(args.output/(name+'_counts.tsv'),sep='\t',index=False,lineterminator='\n')
        for suffix,value in [('qc',normalized(result.qc_summary)),('rejections',result.rejection_stats)]:
            (args.output/(name+'_'+suffix+'.json')).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
        fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
        rs.plots.abundance_distribution(result.counts_df,ax=axes[0],bins=8)
        axes[0].set_ylim(0, max(p.get_height() for p in axes[0].patches)*1.3)
        rs.plots.depth_representation(result.counts_df,ax=axes[1],thresholds=(1,10,20,40,100))
        rs.plots.rejection_breakdown(result.rejection_stats,ax=axes[2])
        axes[0].set_title(name.replace('_',' '));axes[1].set_title('Read-depth representation')
        fig.savefig(args.output/(name+'_qc.png'),dpi=120,metadata={'Software':'READsparrow0.12.1'});plt.close(fig)
    check=verify(args.output)
    (args.output/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    meta={'readsparrow':rs.__version__,'python':platform.python_version(),'platform':platform.platform(),
          'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','pandas','matplotlib']}}
    (args.output/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(check,indent=2))

if __name__=='__main__':
    main()
