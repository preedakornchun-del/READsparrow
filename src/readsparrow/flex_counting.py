"""Full-reference, identifiable-sequence counting; never allocate ambiguous IDs.

Adapter edits are allowed, spacer edits are not. Downstream scaffold coverage
is not required. Sequence groups are measurement units, NOT independent guides
or biological control blocks. Old counting/inference modules are unchanged.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .counting import fastq_records, reverse_complement
from .io import read_rows, write_tsv, write_json


@dataclass
class GroupLibrary:
    ids: list[str]
    genes: list[str]
    sequences: list[bytes]

    def __post_init__(self):
        if not self.ids or len(self.ids)!=len(self.genes) or len(self.ids)!=len(self.sequences):
            raise ValueError('Nonempty library with matching ID/gene/sequence columns required')
        if len(set(self.ids))!=len(self.ids) or any(not x for x in self.ids+self.genes):
            raise ValueError('Guide IDs must be unique; ID/gene annotations must be nonempty')
        if len({len(s) for s in self.sequences})!=1 or not self.sequences[0] or any(set(s)-set(b'ACGT') for s in self.sequences):
            raise ValueError('Uniform, nonempty A/C/G/T spacers required')
        by_sequence=defaultdict(list)
        for i,s in enumerate(self.sequences):by_sequence[s].append(i)
        # Content-derived groups are stable under reference row reordering.
        self.group_sequences=sorted(by_sequence)
        self.members=[by_sequence[s] for s in self.group_sequences]
        self.group_ids=['SEQ_'+hashlib.sha256(s).hexdigest() for s in self.group_sequences]
        self.lookup={s:i for i,s in enumerate(self.group_sequences)}
        self.length=len(self.sequences[0])


def load_group_library(path):
    header,rows=read_rows(path)
    aliases=[('sgRNA','ID','guide_id','UID'),('Gene','gene_symbol','gene','gene_id'),('Sequence','seq','sequence')]
    cols=[next((c for c in options if c in header),None) for options in aliases]
    if any(c is None for c in cols):raise ValueError('Library needs ID, gene and sequence columns')
    try:return GroupLibrary([r[cols[0]] for r in rows],[r[cols[1]] for r in rows],
        [r[cols[2]].upper().encode('ascii') for r in rows])
    except UnicodeEncodeError as exc:raise ValueError('Non-ASCII spacer sequence') from exc


def _bounded_end_costs(adapter, text, k):
    """Banded full-adapter Levenshtein costs for all admissible text ends."""
    m,n=len(adapter),len(text);inf=k+1
    prev=[j if j<=k else inf for j in range(n+1)]
    for i,a in enumerate(adapter,1):
        current=[inf]*(n+1)
        if i<=k:current[0]=i
        for j in range(max(1,i-k),min(n,i+k)+1):
            cost=0 if a==78 or a==text[j-1] else 1
            current[j]=min(prev[j]+1,current[j-1]+1,prev[j-1]+cost,inf)
        prev=current
    return [(j,prev[j]) for j in range(max(1,m-k),min(n,m+k)+1) if prev[j]<=k]


class AdapterExtractor:
    def __init__(self,library,adapter,max_edits=2,orientation='forward'):
        if orientation not in ('forward','reverse'):raise ValueError('Explicit forward/reverse orientation required')
        if not 0<=max_edits<=2:raise ValueError('max_adapter_edits must be 0, 1 or 2')
        if len(adapter)<12 or sum(x!=78 for x in adapter)<12 or set(adapter)-set(b'ACGTN'):
            raise ValueError('Adapter requires >=12 informative A/C/G/T bases; only A/C/G/T/N accepted')
        self.lib,self.adapter,self.k,self.orientation=library,adapter,max_edits,orientation
        # Pick disjoint literal seeds. If wildcards prevent a complete seed set,
        # exhaustive bounded starts are used, not a sensitivity-losing heuristic.
        seeds=[];m=len(adapter)
        for b in range(max_edits+1):
            lo=b*m//(max_edits+1);hi=(b+1)*m//(max_edits+1)
            runs=[];start=lo
            for j in range(lo,hi+1):
                if j==hi or adapter[j]==78:
                    if j>start:runs.append((j-start,start,adapter[start:j]))
                    start=j+1
            if runs:
                length,offset,seed=max(runs)
                if length>=3:seeds.append((offset,seed))
        self.seeds=seeds if len(seeds)==max_edits+1 else None
        literal_runs=[];start=0
        for j in range(m+1):
            if j==m or adapter[j]==78:
                if j>start:literal_runs.append((j-start,start,adapter[start:j]))
                start=j+1
        _,self.exact_offset,self.exact_seed=max(literal_runs)

    def placement(self,read):
        a=self.adapter;m=len(a);exact=set()
        pos=read.find(self.exact_seed)
        while pos>=0:
            start=pos-self.exact_offset
            if start>=0 and start+m<=len(read):
                region=read[start:start+m]
                if all(x==78 or x==y for x,y in zip(a,region)):exact.add(start+m)
            pos=read.find(self.exact_seed,pos+1)
        if exact:return exact,0
        if not self.k:return set(),None
        starts=set()
        if self.seeds is None:
            starts.update(range(max(0,len(read)-(m-self.k)+1)))
        else:
            for offset,seed in self.seeds:
                pos=read.find(seed)
                while pos>=0:
                    starts.update(s for s in range(pos-offset-self.k,pos-offset+self.k+1)
                        if 0<=s<=len(read)-(m-self.k))
                    pos=read.find(seed,pos+1)
        best=self.k+1;ends=set()
        for start in sorted(starts):
            for length,cost in _bounded_end_costs(a,read[start:start+m+self.k],self.k):
                if cost<best:best=cost;ends={start+length}
                elif cost==best:ends.add(start+length)
        return (ends,best) if ends else (set(),None)

    def match(self,read):
        if self.orientation=='reverse':read=reverse_complement(read)
        ends,edits=self.placement(read)
        if not ends:return None,'adapter_not_found',None
        if len(ends)!=1:return None,'ambiguous_adapter_placement',edits
        end=next(iter(ends))
        if len(read)-end<self.lib.length:return None,'incomplete_spacer',edits
        index=self.lib.lookup.get(read[end:end+self.lib.length])
        if index is None:return None,'spacer_no_exact_reference_match',edits
        return index,('assigned_sequence_group' if len(self.lib.members[index])>1 else 'assigned_unique_id'),edits


def count_flexible(library,path,sample,adapter,max_edits=2,orientation='forward',max_reads=None):
    extractor=AdapterExtractor(library,adapter,max_edits,orientation)
    counts=np.zeros(len(library.group_ids),dtype=np.int64)
    categories=Counter();edit_counts=Counter();start=time.perf_counter();cpu=time.process_time()
    for read in fastq_records(path,max_reads):
        group,reason,edits=extractor.match(read);categories[reason]+=1
        if group is not None:counts[group]+=1;edit_counts[str(edits)]+=1
    total=sum(categories.values())
    if not total:raise ValueError('FASTQ contains no records')
    resolved=categories['assigned_unique_id'];ambiguous=categories['assigned_sequence_group']
    metrics={'sample':sample,'total_reads':total,'matched_sequence_reads':resolved+ambiguous,
        'resolved_id_reads':resolved,'duplicate_sequence_reads':ambiguous,
        'matched_sequence_percent':100*(resolved+ambiguous)/total,'resolved_id_percent':100*resolved/total,
        'categories':dict(categories),'assigned_adapter_edit_counts':dict(edit_counts),
        'reference_rows':len(library.ids),'distinct_sequences':len(library.group_ids),
        'duplicate_sequence_groups':sum(len(x)>1 for x in library.members),
        'duplicate_reference_rows':sum(len(x) for x in library.members if len(x)>1),
        'adapter':adapter.decode(),'max_adapter_edits':max_edits,'orientation':orientation,
        'spacer_length':library.length,'spacer_edit_rescue':False,'right_anchor_required':False,
        'ambiguous_id_allocation':'none; measured once per sequence group',
        'group_counts_compatible_with_existing_hit_caller':False,
        'loop_wall_s':time.perf_counter()-start,'loop_cpu_s':time.process_time()-cpu}
    assert resolved+ambiguous==int(counts.sum())
    return counts,metrics


def save_flexible(out,library,counts,metrics,sample):
    genes=Counter()
    def sequence_rows():
        for g,(seq,members) in enumerate(zip(library.group_sequences,library.members)):
            value=int(counts[g]);gene_set=sorted({library.genes[i] for i in members})
            if len(gene_set)==1:genes[gene_set[0]]+=value
            yield {'sequence_group':library.group_ids[g],'Sequence':seq.decode(),
                'member_ids':';'.join(sorted(library.ids[i] for i in members)),
                'genes':';'.join(gene_set),'n_ids':len(members),
                'gene_identifiable':len(gene_set)==1,'count':value}
    def guide_rows(resolved_only=False):
        # Preserve original reference order and membership, streaming output.
        for i,seq in enumerate(library.sequences):
            g=library.lookup[seq];identifiable=len(library.members[g])==1;value=int(counts[g])
            if resolved_only:
                if identifiable:yield {'sgRNA':library.ids[i],'Gene':library.genes[i],sample:value}
            else:yield {'sgRNA':library.ids[i],'Gene':library.genes[i],
                'Sequence':seq.decode(),'sequence_group':library.group_ids[g],
                'id_identifiable':identifiable,'resolved_count':value if identifiable else '',
                'compatible_group_reads':value}
    write_tsv(out/'sequence_counts.tsv',['sequence_group','Sequence','member_ids','genes','n_ids','gene_identifiable','count'],sequence_rows())
    write_tsv(out/'guide_audit.tsv',['sgRNA','Gene','Sequence','sequence_group','id_identifiable','resolved_count','compatible_group_reads'],guide_rows())
    write_tsv(out/'resolved_counts.tsv',['sgRNA','Gene',sample],guide_rows(True))
    write_tsv(out/'identifiable_gene_counts.tsv',['gene','count'],
        [{'gene':g,'count':n} for g,n in sorted(genes.items())])
    write_json(out/'count_metrics.json',metrics)
    (out/'COUNTING_NOTES.txt').write_text(
        'The full input reference was indexed without dropping any row.\n'
        'sequence_counts.tsv measures each observed sequence group ONCE.\n'
        'guide_audit.tsv retains every original ID: ambiguous resolved_count is BLANK, not zero.\n'
        'compatible_group_reads is contextual and MUST NOT be summed across member IDs.\n'
        'resolved_counts.tsv is an explicit identifiable-ID subset, NOT a full-reference matrix.\n'
        'Group counts are not independent guides or neutral-control blocks. Existing hit callers\n'
        'must not consume sequence groups as if they were ordinary independent sgRNAs.\n'
        'Gene counts include only groups whose members all have the same gene annotation.\n'
        'Adapter errors are allowed; spacer errors are not corrected. Missing adapters and tied\n'
        'best adapter endpoints are rejected. No downstream scaffold bases are fabricated.\n')
