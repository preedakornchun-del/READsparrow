import csv
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
from readsparrow.counting import reverse_complement
from readsparrow.flex_counting import GroupLibrary,AdapterExtractor,_bounded_end_costs,count_flexible,save_flexible
from readsparrow.io import load_counts

A=b'TCTTGTGGAAAGGACGAAACACCN'
ACT=A[:-1]+b'G'
G=b'ACGTGCACTGATCGATGACA'

def exhaustive(a,text,k):
    best=k+1;ends=set()
    for start in range(len(text)):
        for end in range(start+max(1,len(a)-k),min(len(text),start+len(a)+k)+1):
            s=text[start:end];prev=list(range(len(s)+1))
            for i,x in enumerate(a,1):
                row=[i]
                for j,y in enumerate(s,1):row.append(min(prev[j]+1,row[j-1]+1,prev[j-1]+int(x!=78 and x!=y)))
                prev=row
            cost=prev[-1]
            if cost<best:best=cost;ends={end}
            elif cost==best:ends.add(end)
    return (ends,best) if best<=k else (set(),None)

class FlexibleTests(unittest.TestCase):
    def setUp(self):self.lib=GroupLibrary(['a','alias','b'],['G1','G2','G3'],[G,G,b'T'*20])
    def test_duplicate_groups_stable_no_id_allocation(self):
        reordered=GroupLibrary(['b','alias','a'],['G3','G2','G1'],[b'T'*20,G,G])
        self.assertEqual(self.lib.group_ids,reordered.group_ids)
        self.assertEqual(len(self.lib.ids),3);self.assertEqual(len(self.lib.group_ids),2)
        index,reason,_=AdapterExtractor(self.lib,A).match(b'NCGAA'+ACT+G+b'G')
        self.assertEqual(self.lib.group_sequences[index],G);self.assertEqual(reason,'assigned_sequence_group')
    def test_complete_50bp_and_no_right_base(self):
        ext=AdapterExtractor(self.lib,A)
        for extra in [b'',b'G',b'T',b'GTTTT']:
            self.assertIsNotNone(ext.match(b'NCGAA'+ACT+G+extra)[0])
        self.assertEqual(len(b'NCGAA'+ACT+G+b'G'),50)
    def test_orientation(self):
        read=b'NCGAA'+ACT+G+b'G'
        self.assertEqual(AdapterExtractor(self.lib,A).match(read)[0],
            AdapterExtractor(self.lib,A,orientation='reverse').match(reverse_complement(read))[0])
    def test_spacer_substitution_not_rescued(self):
        seq=G[:10]+b'C'+G[11:]
        self.assertNotEqual(seq,G)
        self.assertEqual(AdapterExtractor(self.lib,A).match(ACT+seq)[1],'spacer_no_exact_reference_match')
    def test_incomplete_reject_and_no_adapter(self):
        e=AdapterExtractor(self.lib,A)
        self.assertEqual(e.match(ACT+G[:-1])[1],'incomplete_spacer')
        self.assertEqual(e.match(G)[1],'adapter_not_found')
    def test_ambiguous_placements_reject(self):
        self.assertEqual(AdapterExtractor(self.lib,A).match(ACT+G+ACT+b'T'*20)[1],'ambiguous_adapter_placement')
    def test_adapter_errors_against_exhaustive_oracle(self):
        rng=random.Random(2811001);e=AdapterExtractor(self.lib,A)
        reads=[ACT+G]
        for _ in range(90):
            adapter=bytearray(ACT)
            for j in range(rng.randrange(1,3)):
                pos=rng.randrange(len(adapter));typ=rng.choice(['sub','del','ins'])
                if typ=='del':del adapter[pos]
                elif typ=='ins':adapter[pos:pos]=rng.choice([b'A',b'C',b'G',b'T'])
                else:adapter[pos]=next(x for x in b'ACGT' if x!=adapter[pos])
            reads.append(b'CC'+bytes(adapter)+G)
        for read in reads:
            with self.subTest(read=read):self.assertEqual(e.placement(read),exhaustive(A,read,2))
    def test_wildcard_interior_and_absent_seeds_oracle(self):
        for adapter in [b'ACGTNNACGTACGTACGT',b'ACGTACGTACGT',b'ACGNNNTACGNNNTACGTACGT']:
            actual=adapter.replace(b'N',b'C');e=AdapterExtractor(self.lib,adapter)
            for read in [actual+G,actual[:5]+b'T'+actual[6:]+G,b'GG'+actual[1:]+G]:
                self.assertEqual(e.placement(read),exhaustive(adapter,read,2))
    def test_export_no_double_count_or_fake_zero_and_downstream_guard(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);fq=p/'a.fastq'
            reads=[ACT+G,ACT+G,ACT+b'T'*20,ACT+G[:-1]]
            fq.write_text(''.join('@r'+str(i)+'\n'+s.decode()+'\n+\n'+'I'*len(s)+'\n' for i,s in enumerate(reads)))
            values,m=count_flexible(self.lib,fq,'sample',A)
            self.assertEqual(m['total_reads'],4);self.assertEqual(int(values.sum()),3)
            self.assertEqual(m['resolved_id_reads'],1);self.assertEqual(m['duplicate_sequence_reads'],2)
            save_flexible(p,self.lib,values,m,'sample')
            with (p/'guide_audit.tsv').open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual(len(rows),3)
            self.assertEqual([r['resolved_count'] for r in rows if r['sgRNA'] in ['a','alias']],['',''])
            self.assertEqual(int(load_counts(p/'resolved_counts.tsv').counts.sum()),1)
            with self.assertRaises(ValueError):load_counts(p/'sequence_counts.tsv')
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):GroupLibrary(['x','x'],['G','G'],[G,G])
        for a,k in [(b'AAAAC',2),(A,3),(b'NNNNNNNNNNNN',2)]:
            with self.assertRaises(ValueError):AdapterExtractor(self.lib,a,k)

if __name__=='__main__':unittest.main()
