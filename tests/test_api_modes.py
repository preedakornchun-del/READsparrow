import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import readsparrow as rs
from readsparrow.counting import ExactExtractor, Library, count_fastq, reverse_complement
from readsparrow.orientation import OrientationWarning, TolerantExtractor, anchor_variants, count_extended

G=b'ACGTGCACTGATCGATGACA'

def write_fastq(path,reads):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wb') as handle:
        for i,sequence in enumerate(reads):
            handle.write(b'@r%d\n'%i+sequence+b'\n+\n'+b'I'*len(sequence)+b'\n')
    return path

def lib(guide=G):
    return Library(['g'],['GENE'],[guide])

def frame(guide=G):
    return pd.DataFrame({'sgRNA':['g'],'Gene':['GENE'],'Sequence':[guide.decode()]})

@pytest.mark.parametrize('length',[19,20])
@pytest.mark.parametrize('orientation',['forward','reverse'])
def test_exact_baseline_vectors_and_rejections(tmp_path,length,orientation):
    guide=(b'AAAGTTTT'+b'C'*12)[:length]
    reads=[b'CACCG'+guide+b'GTTTT',b'CACCGAA'+guide+b'GTTTT',b'GGG',
           b'CACCG'+guide,b'CACCG'+b'T'*length+b'GTTTT']
    if orientation=='reverse': reads=[reverse_complement(s) for s in reads]
    path=write_fastq(tmp_path/'reads.fastq.gz',reads)
    baseline,expected=count_fastq(lib(guide),path,'count',orientation)
    result=rs.count(path,frame(guide),orientation=orientation)
    np.testing.assert_array_equal(result.counts_df['count'],baseline.counts[:,0])
    assert result.rejection_stats==expected['categories']
    assert result.rejection_stats=={'assigned_exact':2,'no_left_anchor':1,
        'no_usable_right_anchor':1,'anchor_region_no_exact_reference_match':1}
    for key,value in expected.items():
        if not key.startswith('loop_'): assert result.qc_summary[key]==value

@pytest.mark.parametrize('orientation',['forward','reverse'])
@pytest.mark.parametrize('left,right',[(b'CATCG',b'GTTTT'),(b'CACCG',b'GTTTA'),(b'CATCG',b'GTTTA')])
def test_one_anchor_substitution_each(tmp_path,orientation,left,right):
    read=left+G+right
    if orientation=='reverse': read=reverse_complement(read)
    path=write_fastq(tmp_path/'reads.fastq',[read])
    assert rs.count(path,frame(),orientation=orientation).counts_df['count'].sum()==0
    assert rs.count(path,frame(),orientation=orientation,anchor_mismatch=1).counts_df['count'].sum()==1

def test_variant_set_no_indels_and_exact_guide_only(tmp_path):
    assert len(anchor_variants(b'CACCG'))==16
    altered=G[:8]+b'A'+G[9:]
    assert altered!=G
    reads=[b'CAATG'+G+b'GTTTT',b'CACCG'+altered+b'GTTTT']
    result=rs.count(write_fastq(tmp_path/'x.fastq',reads),frame(),orientation='forward',anchor_mismatch=1)
    assert result.counts_df['count'].sum()==0

def test_internal_motif_and_first_window_in_tolerant_mode():
    guide=b'AAAGTTTT'+b'C'*12
    ext=TolerantExtractor(lib(guide),'forward',b'CACCG',b'GTTTT')
    assert ext.match(b'CATCG'+guide+b'GTTTA')==(0,'assigned_exact')
    library=Library(['a','b'],['A','B'],[b'A'*20,b'C'*20])
    assert TolerantExtractor(library,'forward',b'CACCG',b'GTTTT').match(
        b'CATCG'+b'A'*20+b'C'*20+b'GTTTA')==(0,'assigned_exact')

def test_auto_reverse_no_prefix_double_count(tmp_path):
    path=write_fastq(tmp_path/'x.fastq',[reverse_complement(b'CACCG'+G+b'GTTTT')]*7)
    result=rs.count(path,frame())
    assert result.qc_summary['orientation']=='reverse'
    assert result.qc_summary['orientation_sniff']['records']==7
    assert result.counts_df['count'].sum()==7
    a,b,c=result
    assert a is result.counts_df and b is result.qc_summary and c is result.rejection_stats

def test_auto_max_reads_and_ambiguous_warning(tmp_path):
    f=b'CACCG'+G+b'GTTTT'; r=reverse_complement(f)
    path=write_fastq(tmp_path/'x.fastq',[f,r,f,r])
    with pytest.warns(OrientationWarning): result=rs.count(path,frame(),max_reads=2)
    assert result.qc_summary['orientation']=='forward'
    assert result.qc_summary['total_reads']==2
    assert result.counts_df['count'].sum()==1
    with pytest.warns(OrientationWarning):
        empty_match=rs.count(write_fastq(tmp_path/'n.fastq',[b'NNNN']),frame())
    assert empty_match.qc_summary['mapped_reads']==0

def test_mixed_rc_collision_and_one_count_per_read(tmp_path):
    rc=reverse_complement(G)
    reference=pd.DataFrame({'sgRNA':['g','rc'],'Gene':['A','B'],'Sequence':[G.decode(),rc.decode()]})
    f=b'CACCG'+G+b'GTTTT'; other=b'CACCG'+rc+b'GTTTT'
    reads=[f,reverse_complement(f),other,reverse_complement(other),f+reverse_complement(other)]
    result=rs.count(write_fastq(tmp_path/'x.fastq',reads),reference,orientation='mixed')
    assert result.counts_df['count'].tolist()==[3,2]
    assert sum(result.rejection_stats.values())==5

def test_unique_policy_retained(tmp_path):
    library=Library(['a','b'],['A','B'],[b'A'*20,b'C'*20])
    path=write_fastq(tmp_path/'x.fastq',[b'CACCG'+b'A'*20+b'C'*20+b'GTTTT'])
    _,m=count_extended(library,path,policy='unique')
    assert m['categories']=={'ambiguous_multiple_guides':1}
    _,m=count_extended(library,path,policy='unique',anchor_mismatch=1)
    assert m['categories']=={'ambiguous_multiple_guides':1}

@pytest.mark.parametrize('kwargs',[{'anchor_mismatch':2},{'anchor_mismatch':True},{'anchor_mismatch':1.0},
    {'orientation':'bad'},{'left_anchor':''},{'max_reads':0},{'sample':'Gene'}])
def test_invalid_options(tmp_path,kwargs):
    with pytest.raises(ValueError): rs.count(write_fastq(tmp_path/'x.fastq',[b'CACCG'+G+b'GTTTT']),frame(),**kwargs)

def test_bad_reference_and_truncated_fastq(tmp_path):
    path=write_fastq(tmp_path/'x.fastq',[b'CACCG'+G+b'GTTTT'])
    with pytest.raises(ValueError): rs.count(path,pd.concat([frame(),frame()]),orientation='forward')
    path.write_bytes(b'@broken\nACGT\n+\n')
    with pytest.raises(ValueError): rs.count(path,frame(),orientation='forward')

def test_core_cli_has_no_scipy_import(tmp_path):
    source='''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'scipy','pysam','Bio','cutadapt','pandas','matplotlib'}:
   raise ImportError('blocked '+fullname,name=fullname)
sys.meta_path.insert(0,Block())
import readsparrow
from readsparrow.cli import parser
assert parser().parse_args(['qc','--counts','c','--output','o']).command=='qc'
assert 'scipy' not in sys.modules and 'pandas' not in sys.modules
'''
    subprocess.run([sys.executable,'-B','-c',source],check=True)

def test_reverse_internal_downstream_motif(tmp_path):
    guide=b'AAACACCG'+b'C'*12
    read=reverse_complement(b'CACCG'+guide+b'GTTTT')
    assert b'CGGTG' in reverse_complement(guide)
    path=write_fastq(tmp_path/'x.fastq',[read])
    for mismatch in [0,1]:
        result=rs.count(path,frame(guide),orientation='reverse',anchor_mismatch=mismatch)
        assert result.counts_df['count'].sum()==1

def test_auto_sniff_cap(tmp_path):
    f=b'CACCG'+G+b'GTTTT'
    path=write_fastq(tmp_path/'x.fastq',[f]*20000+[reverse_complement(f)])
    result=rs.count(path,frame())
    assert result.qc_summary['orientation_sniff']['records']==20000
    assert result.qc_summary['total_reads']==20001
    assert result.counts_df['count'].sum()==20000

def test_tolerant_regex_against_independent_hamming_scan():
    import random
    rng=random.Random(17112026)
    for orientation in ['forward','reverse']:
        extractor=TolerantExtractor(lib(),orientation,b'CACCG',b'GTTTT')
        def first_match(read,anchor,start):
            for pos in range(start,len(read)-len(anchor)+1):
                window=read[pos:pos+len(anchor)]
                if set(window)<=set(b'ACGT') and sum(a!=b for a,b in zip(window,anchor))<=1:
                    return pos
            return -1
        for _ in range(300):
            left=bytearray(extractor.left); right=bytearray(extractor.right)
            for anchor in [left,right]:
                for pos in rng.sample(range(5),rng.randrange(3)):
                    anchor[pos]=rng.choice([b for b in b'ACGT' if b!=anchor[pos]])
            guide=G if orientation=='forward' else reverse_complement(G)
            if rng.random()<.25: guide=b'N'+guide[1:]
            prefix=bytes(rng.choice(b'ACGTN') for _ in range(rng.randrange(15)))
            read=prefix+bytes(left)+guide+bytes(right)
            def strict_oracle():
                p=read.find(extractor.left)
                if p<0: return None
                end=read.find(extractor.right,p+5+20)
                if end<0: return None
                canonical=G if orientation=='forward' else reverse_complement(G)
                return 0 if any(read[i:i+20]==canonical for i in range(p+5,end-20+1)) else None
            p=first_match(read,extractor.left,0)
            expected=(None,'no_left_anchor')
            saw_right=False
            if p>=0:
                expected=(None,'no_usable_right_anchor')
                while p>=0:
                    start=p+5; end=first_match(read,extractor.right,start+20)
                    if end>=0:
                        saw_right=True
                        for offset in range(start,end-20+1):
                            if read[offset:offset+20]==(G if orientation=='forward' else reverse_complement(G)):
                                expected=(0,'assigned_exact'); break
                    if expected[0] is not None: break
                    p=first_match(read,extractor.left,p+1)
                if expected[0] is None and saw_right:
                    expected=(None,'anchor_region_no_exact_reference_match')
            if strict_oracle() is not None: expected=(0,'assigned_exact')
            assert extractor.match(read)==expected

def test_tolerant_cannot_displace_a_valid_strict_cassette(tmp_path):
    guide=b'TAGTCGCTCCGGGGAGACGT'
    read=b'ACACACAC'+b'CACCG'+guide+b'GTTTT'
    for orientation in ['forward','reverse']:
        seq=read if orientation=='forward' else reverse_complement(read)
        path=write_fastq(tmp_path/(orientation+'.fastq'),[seq])
        for mismatch in [0,1]:
            assert rs.count(path,frame(guide),orientation=orientation,anchor_mismatch=mismatch).counts_df['count'].sum()==1

def test_cli_new_modes_and_legacy_dependency_error(tmp_path):
    from readsparrow.cli import main
    reference=tmp_path/'lib.tsv'; frame().to_csv(reference,sep='\t',index=False)
    read=b'CATCG'+G+b'GTTTA'
    path=write_fastq(tmp_path/'x.fastq',[read,reverse_complement(read)])
    assert main(['count','--library',str(reference),'--fastq',str(path),'--sample','S',
                 '--orientation','mixed','--anchor-mismatch','1','--output',str(tmp_path/'out')])==0
    assert pd.read_csv(tmp_path/'out/counts.tsv',sep='\t')['S'].sum()==2
    source='''import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]=='scipy': raise ImportError('missing scipy',name=fullname)
sys.meta_path.insert(0,Block())
from readsparrow.cli import main
main(['test','--counts','c','--samples','s','--treatment','T','--control','C','--experimental','--output','unused'])
'''
    p=subprocess.run([sys.executable,'-B','-c',source],capture_output=True,text=True)
    assert p.returncode==2 and 'readsparrow[legacy]' in p.stderr
