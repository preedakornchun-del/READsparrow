"""Observed variants and conservative candidate annotations, never repaired counts.

Exact counting reuses the frozen v0.11 adapter placement. References remain
complete. Candidate C is the unchanged audit policy, generalized to uniform
spacer length. Its open-world precision is NOT established.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import gzip
import hashlib
from pathlib import Path
import time
import numpy as np
from .counting import reverse_complement
from .flex_counting import AdapterExtractor, save_flexible
from .io import write_json, write_tsv

_BASE={65:0,67:1,71:2,84:3}
def encode(sequence):
    code=0
    for base in sequence:code=(code<<2)|_BASE[base]
    return code

class VariantIndex:
    """Complete Hamming<=2 via three disjoint blocks; no edit-index expansion."""
    def __init__(self,library):
        if library.length<3:raise ValueError('Variant reporting requires spacer length >=3')
        self.lib=library;self.length=library.length
        self.codes=[encode(s) for s in library.group_sequences]
        self.mask=sum(1<<(2*p) for p in range(self.length));self.blocks=[]
        for lo,hi in zip([0,self.length//3,2*self.length//3],[self.length//3,2*self.length//3,self.length]):
            shift=2*(self.length-hi);mask=(1<<(2*(hi-lo)))-1;index=defaultdict(list)
            for g,code in enumerate(self.codes):index[(code>>shift)&mask].append(g)
            self.blocks.append((shift,mask,dict(index)))
        self.neighbours=lru_cache(maxsize=100000)(self._neighbours)

    def _neighbours(self,query):
        if len(query)!=self.length or set(query)-set(b'ACGT'):return (),()
        code=encode(query);candidates=set();one=[];two=[]
        for shift,mask,index in self.blocks:candidates.update(index.get((code>>shift)&mask,()))
        for g in candidates:
            x=code^self.codes[g];distance=((x|(x>>1))&self.mask).bit_count()
            if distance==1:one.append(g)
            elif distance==2:two.append(g)
        return tuple(sorted(one)),tuple(sorted(two))

    def deletion_groups(self,query):
        if len(query)!=self.length-1 or set(query)-set(b'ACGT'):return set()
        groups=set()
        # Inserting one base into the observed L-1 string enumerates all
        # references with a one-base deletion signature, including homopolymers.
        for p in range(self.length):
            for base in (b'A',b'C',b'G',b'T'):
                g=self.lib.lookup.get(query[:p]+base+query[p:])
                if g is not None:groups.add(g)
        return groups

    def insertion_groups(self,query):
        if len(query)!=self.length+1 or set(query)-set(b'ACGT'):return set()
        return {g for p in range(self.length+1)
            if (g:=self.lib.lookup.get(query[:p]+query[p+1:])) is not None}

    def candidate(self,read,quality,end,edits,one,two):
        """Quality gate first is logically identical to the inherited C rule."""
        if len(one)!=1 or two or edits!=0:return None
        g=one[0];query=read[end:end+self.length];reference=self.lib.group_sequences[g]
        p=next(i for i,(a,b) in enumerate(zip(query,reference)) if a!=b)
        qs=[q-33 for q in quality[end:end+self.length]]
        if qs[p]>20 or min(q for i,q in enumerate(qs) if i!=p)<20:return None
        alternatives=set()
        for offset in [-3,-2,-1,1,2,3]:
            start=end+offset
            if start>=0 and start+self.length<=len(read):
                other=self.lib.lookup.get(read[start:start+self.length])
                if other is not None:alternatives.add(other)
        other=self.lib.lookup.get(reverse_complement(query))
        if other is not None:alternatives.add(other)
        alternatives.update(self.deletion_groups(read[end:end+self.length-1]))
        alternatives.update(self.insertion_groups(read[end:end+self.length+1]))
        if alternatives-{g}:return None
        return g

def quality_fastq(path,max_reads=None):
    if max_reads is not None and max_reads<1:raise ValueError('max_reads must be positive')
    path=Path(path);opener=gzip.open if path.suffix.lower()=='.gz' else open
    with opener(path,'rb') as f:
        i=0
        while max_reads is None or i<max_reads:
            header=f.readline()
            if not header:return
            sequence=f.readline();plus=f.readline();quality=f.readline()
            seq=sequence.rstrip(b'\r\n').upper();q=quality.rstrip(b'\r\n')
            if not header.startswith(b'@') or not plus.startswith(b'+') or not seq or len(seq)!=len(q):
                raise ValueError('Malformed four-line FASTQ at record '+str(i+1))
            if set(seq)-set(b'ACGTN') or any(x<33 or x>126 for x in q):
                raise ValueError('Invalid DNA or Phred+33 encoding at record '+str(i+1))
            yield i,header.rstrip(b'\r\n'),seq,q;i+=1

def count_variants(library,path,sample,adapter,annotation_path,max_edits=2,orientation='forward',max_reads=None):
    extractor=AdapterExtractor(library,adapter,max_edits,orientation)
    index=VariantIndex(library);counts=np.zeros(len(library.group_ids),dtype=np.int64)
    categories=Counter();edits_hist=Counter();partition=Counter();candidates=Counter();variants={}
    start=time.perf_counter();cpu=time.process_time()
    with gzip.open(annotation_path,'wt') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['sample','read_index','read_id','observed_variant',
            'possible_parent_group','mismatch_position_1based','reported_mismatch_Q','not_confirmed_recovery'])
        for i,header,read,q in quality_fastq(path,max_reads):
            if orientation=='reverse':read=reverse_complement(read);q=q[::-1]
            ends,edits=extractor.placement(read)
            if not ends:categories['adapter_not_found']+=1;continue
            if len(ends)!=1:categories['ambiguous_adapter_placement']+=1;continue
            end=next(iter(ends))
            if end+library.length>len(read):categories['incomplete_spacer']+=1;continue
            query=read[end:end+library.length];g=library.lookup.get(query)
            if g is not None:
                counts[g]+=1;edits_hist[str(edits)]+=1
                categories['assigned_sequence_group' if len(library.members[g])>1 else 'assigned_unique_id']+=1
                continue
            categories['spacer_no_exact_reference_match']+=1
            one,two=index.neighbours(query)
            if b'N' in query:label='contains_N'
            elif one:label='H1_unique' if len(one)==1 else 'H1_tied'
            elif two:label='H2_unique' if len(two)==1 else 'H2_tied'
            else:label='no_Hamming_neighbour_within2'
            partition[label]+=1
            v=variants.get(query)
            if v is None:
                # One row per actual observed string, never one row per parent.
                v={'reads':0,'one':one,'two':two,'class':label,'Q30':0,'qsum':0,'candidate':0,'mismatch_qsum':0,'mismatch_Q30':0}
                variants[query]=v
            v['reads']+=1;qs=[x-33 for x in q[end:end+library.length]]
            v['Q30']+=min(qs)>=30;v['qsum']+=sum(qs)
            if len(one)==1:
                parent=one[0];p=next(j for j,(a,b) in enumerate(zip(query,library.group_sequences[parent])) if a!=b)
                v['mismatch_qsum']+=qs[p];v['mismatch_Q30']+=qs[p]>=30
            candidate=index.candidate(read,q,end,edits,one,two)
            if candidate is not None:
                candidates[candidate]+=1;v['candidate']+=1
                w.writerow([sample,i,header.decode('ascii'),query.decode(),library.group_ids[candidate],p+1,qs[p],True])
    total=sum(categories.values())
    if not total:raise ValueError('FASTQ contains no records')
    resolved=categories['assigned_unique_id'];duplicate=categories['assigned_sequence_group']
    metrics={'sample':sample,'total_reads':total,'matched_sequence_reads':resolved+duplicate,
        'resolved_id_reads':resolved,'duplicate_sequence_reads':duplicate,'matched_sequence_percent':100*(resolved+duplicate)/total,
        'resolved_id_percent':100*resolved/total,'categories':dict(categories),'assigned_adapter_edit_counts':dict(edits_hist),
        'reference_rows':len(library.ids),'distinct_sequences':len(library.group_ids),
        'duplicate_sequence_groups':sum(len(x)>1 for x in library.members),
        'duplicate_reference_rows':sum(len(x) for x in library.members if len(x)>1),
        'adapter':adapter.decode(),'max_adapter_edits':max_edits,'orientation':orientation,'spacer_length':library.length,
        'spacer_edit_rescue':False,'right_anchor_required':False,'ambiguous_id_allocation':'none; measured once per sequence group',
        'group_counts_compatible_with_existing_hit_caller':False,'observed_variant_reads':sum(partition.values()),
        'variant_partition':dict(partition),'distinct_observed_variants':len(variants),
        'quality_supported_candidate_reads':sum(candidates.values()),'candidate_counts_merged':False,
        'candidate_policy':'Inherited audit C; changedQ<=20; otherQ>=20; unique H1; no H2; adapter0; no different-group alternate representation',
        'real_data_true_origin_known':False,'universal_open_world_precision_established':False,
        'loop_wall_s':time.perf_counter()-start,'loop_cpu_s':time.process_time()-cpu}
    assert int(counts.sum())==resolved+duplicate
    assert sum(v['reads'] for v in variants.values())==sum(partition.values())
    return counts,metrics,variants,candidates

def save_variants(out,library,counts,metrics,variants,candidates,sample):
    save_flexible(out,library,counts,metrics,sample)
    columns=['observed_variant_group','observed_variant','reads','partition','all_bases_Q30_reads','mean_spacer_Q',
        'n_best_possible_parent_groups','possible_parent_groups','possible_parent_sequences','possible_parent_genes',
        'unique_H1_parent_exact_reads','variant_fraction_given_unique_H1_parent','H1_mean_mismatch_Q','H1_mismatch_Q30_reads','candidate_C_reads','confirmed_origin']
    def variant_rows():
        for query,v in sorted(variants.items()):
            parents=v['one'] or v['two'];unique=len(v['one'])==1;g=v['one'][0] if unique else None
            n=v['reads'];p=int(counts[g]) if unique else None
            yield dict(zip(columns,['VAR_'+hashlib.sha256(query).hexdigest(),query.decode(),n,v['class'],v['Q30'],v['qsum']/library.length/n,
                len(parents),';'.join(library.group_ids[k] for k in parents),';'.join(library.group_sequences[k].decode() for k in parents),
                ';'.join(sorted({library.genes[i] for k in parents for i in library.members[k]})),
                p if unique else '',n/(n+p) if unique else '',v['mismatch_qsum']/n if unique else '',v['mismatch_Q30'] if unique else '',v['candidate'],False]))
    write_tsv(out/'observed_variant_counts.tsv',columns,variant_rows())
    write_tsv(out/'candidate_group_counts.tsv',['possible_parent_group','possible_parent_sequence','candidate_reads','parent_n_IDs','parent_genes','confirmed_recovery'],
        ({'possible_parent_group':library.group_ids[g],'possible_parent_sequence':library.group_sequences[g].decode(),
          'candidate_reads':n,'parent_n_IDs':len(library.members[g]),'parent_genes':';'.join(sorted({library.genes[i] for i in library.members[g]})),
          'confirmed_recovery':False} for g,n in sorted(candidates.items())))
    write_json(out/'variant_reporting.json',{'candidate_counts_merged':False,'variants_are_observed_sequences_not_confirmed_molecular_variants':True,
        'possible_parent_is_not_proven_origin':True,'duplicate_IDs_allocated':False,'open_world_precision_established':False,
        'sequence_group_counts_are_not_independent_guide_evidence':True,'accepted_errors_can_be_indistinguishable_from_near_reference_off_library_sequences':True})
    (out/'VARIANT_NOTES.txt').write_text('Exact counts remain separate and do not include candidates.\n'
        'Observed variants are distinct FASTQ strings, not confirmed biological variants.\n'
        'Possible reference parents are sequence-compatible hypotheses, not molecular truth.\n'
        'Candidate C reads can be indistinguishable from real near-reference nonlibrary molecules.\n'
        'A recurrent high-quality variant can also contain low-quality reads satisfying C.\n'
        'Do not merge candidate counts or feed variant groups into hit calling by default.\n')
