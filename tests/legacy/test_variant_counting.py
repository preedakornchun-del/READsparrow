import csv
import gzip
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from readsparrow.counting import reverse_complement
from readsparrow.flex_counting import GroupLibrary,AdapterExtractor,count_flexible
from readsparrow.variant_counting import VariantIndex,count_variants,save_variants,quality_fastq

A=b'GACTACGTCTAGTCAGTA'
G=b'ACGTGCACTGATCGATGACA'
def change(seq,pos):return seq[:pos]+bytes([next(b for b in b'ACGT' if b!=seq[pos])])+seq[pos+1:]
def quality(read,pos):return b'I'*pos+b'-'+b'I'*(len(read)-pos-1)

class VariantTests(unittest.TestCase):
    def setUp(self):self.lib=GroupLibrary(['g','alias','t'],['G1','G2','GT'],[G,G,b'T'*20])
    def test_hamming_oracle_many_lengths(self):
        rng=random.Random(2812001)
        for length in [3,19,20,24,31]:
            seqs=sorted({bytes(rng.choices(b'ACGT',k=length)) for _ in range(60)})
            lib=GroupLibrary([str(i) for i in range(len(seqs))],['G']*len(seqs),seqs);index=VariantIndex(lib)
            for seq in seqs[:20]:
                q=change(seq,rng.randrange(length));q=change(q,rng.randrange(length))
                expected=[tuple(g for g,s in enumerate(lib.group_sequences) if sum(a!=b for a,b in zip(q,s))==d) for d in [1,2]]
                self.assertEqual(index.neighbours(q),tuple(expected))
            self.assertEqual(index.neighbours(b'N'*length),((),()))
    def test_indel_oracle(self):
        index=VariantIndex(self.lib)
        for q in [G[:-1],G[1:],G[:7]+G[8:]]:
            expected={g for g,s in enumerate(self.lib.group_sequences) if any(s[:p]+s[p+1:]==q for p in range(20))}
            self.assertEqual(index.deletion_groups(q),expected)
        for q in [G+b'G',b'C'+G,G[:8]+b'T'+G[8:]]:
            expected={g for g,s in enumerate(self.lib.group_sequences) if any(q[:p]+q[p+1:]==s for p in range(21))}
            self.assertEqual(index.insertion_groups(q),expected)
    def test_candidate_quality_gates_and_identical_hard_negative(self):
        index=VariantIndex(self.lib);q=change(G,8);read=A+q+b'GTTTT';one,two=index.neighbours(q)
        low=quality(read,len(A)+8)
        self.assertEqual(index.candidate(read,low,len(A),0,one,two),self.lib.lookup[G])
        self.assertIsNone(index.candidate(read,b'I'*len(read),len(A),0,one,two))
        bad=bytearray(low);bad[len(A)+3]=33
        self.assertIsNone(index.candidate(read,bytes(bad),len(A),0,one,two))
        self.assertIsNone(index.candidate(read,low,len(A),1,one,two))
        # Molecular labels cannot change a deterministic read/quality decision.
        self.assertEqual(index.candidate(read,low,len(A),0,one,two),index.candidate(read,low,len(A),0,one,two))
    def test_H2_competitor_reject(self):
        q=change(G,8);other=change(change(q,3),12)
        lib=GroupLibrary(['a','b'],['A','B'],[G,other]);index=VariantIndex(lib)
        one,two=index.neighbours(q);read=A+q+b'GTTTT'
        self.assertEqual(len(one),1);self.assertEqual(len(two),1)
        self.assertIsNone(index.candidate(read,quality(read,len(A)+8),len(A),0,one,two))
    def test_alternate_shift_RC_and_indel_competitors_reject(self):
        q=change(G,8);read=A+q+b'GTTTT'
        alternatives=[q[1:]+b'G',reverse_complement(q),q[:4]+b'C'+q[4:19],(q+b'G')[:4]+(q+b'G')[5:]]
        for other in alternatives:
            lib=GroupLibrary(['a','b'],['A','B'],[G,other]);idx=VariantIndex(lib);one,two=idx.neighbours(q)
            self.assertIsNone(idx.candidate(read,quality(read,len(A)+8),len(A),0,one,two))
    def test_count_identity_orientation_and_nonallocation(self):
        q=change(G,8);reads=[A+G+b'GTTTT',A+G+b'GTTTT',A+q+b'GTTTT',A+b'T'*20+b'GTTTT',A+G[:-1]]
        qualities=[b'I'*len(r) for r in reads];qualities[2]=quality(reads[2],len(A)+8)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for orientation in ['forward','reverse']:
                fq=root/(orientation+'.fastq');out=root/orientation;out.mkdir()
                with fq.open('wb') as f:
                    for i,(read,qual) in enumerate(zip(reads,qualities)):
                        if orientation=='reverse':read=reverse_complement(read);qual=qual[::-1]
                        f.write(b'@r'+str(i).encode()+b'\n'+read+b'\n+\n'+qual+b'\n')
                old,om=count_flexible(self.lib,fq,'s',A,orientation=orientation)
                counts,m,variants,c=count_variants(self.lib,fq,'s',A,out/'annotations.gz',orientation=orientation)
                np.testing.assert_array_equal(old,counts);self.assertEqual(om['categories'],m['categories'])
                self.assertEqual(sum(c.values()),1);self.assertEqual(int(counts.sum()),3)
                self.assertEqual(m['observed_variant_reads'],1);self.assertFalse(m['candidate_counts_merged'])
                save_variants(out,self.lib,counts,m,variants,c,'s')
                with (out/'guide_audit.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
                self.assertEqual([r['resolved_count'] for r in rows if r['sgRNA'] in ['g','alias']],['',''])
                with gzip.open(out/'annotations.gz','rt') as f:self.assertEqual(len(list(csv.DictReader(f,delimiter='\t'))),1)
    def test_quality_validation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.fastq';p.write_bytes(b'@r\nACGT\n+\n    \n')
            with self.assertRaises(ValueError):list(quality_fastq(p))
            with self.assertRaises(ValueError):list(quality_fastq(p,0))
if __name__=='__main__':unittest.main()
