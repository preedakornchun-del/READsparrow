"""Opt-in anchor substitutions and orientation; guide lookup remains exact."""
from collections import Counter
import re
import time
import warnings
import numpy as np
from .counting import ExactExtractor, count_fastq, fastq_records
from .io import CountTable

class OrientationWarning(UserWarning):
    """The sniffed prefix does not establish one clear orientation."""

def anchor_variants(anchor):
    return tuple(sorted({anchor} | {anchor[:i]+bytes([b])+anchor[i+1:]
        for i,a in enumerate(anchor) for b in b"ACGT" if b != a}))

class TolerantExtractor(ExactExtractor):
    def __init__(self, library, orientation, left, right, policy="legacy", max_extra=32):
        super().__init__(library, orientation, left, right, policy, max_extra)
        self.left_variants, self.right_variants = anchor_variants(self.left), anchor_variants(self.right)
        self.left_pattern = re.compile(b"(?:"+b"|".join(self.left_variants)+b")")
        self.right_pattern = re.compile(b"(?:"+b"|".join(self.right_variants)+b")")

    def match(self, sequence):
        # An earlier approximate motif must not displace an already valid exact
        # cassette. Unique policy still evaluates all compatible identities.
        if self.policy == "legacy":
            strict = ExactExtractor.match(self, sequence)
            if strict[0] is not None:
                return strict
        found = self.left_pattern.search(sequence)
        saw_left = saw_right = saw_long = False
        matches = set()
        while found is not None:
            saw_left = True
            start = found.end()
            # Never stop at a downstream motif inside the guide.
            right = self.right_pattern.search(sequence, start+self.length)
            if right is not None:
                saw_right = True
                if self.policy == "legacy" or right.start()-start <= self.length+self.max_extra:
                    for offset in range(start, right.start()-self.length+1):
                        index = self.lookup.get(sequence[offset:offset+self.length])
                        if index is not None:
                            if self.policy == "legacy":
                                return index, "assigned_exact"
                            matches.add(index)
                else:
                    saw_long = True
            # An approximate upstream motif can occur incidentally in the flank.
            # Continue until the first reference-valid window, never count twice.
            found = self.left_pattern.search(sequence, found.start()+1)
        if len(matches)>1:
            return None, "ambiguous_multiple_guides"
        if matches:
            return next(iter(matches)), "assigned_exact"
        return None, ("no_left_anchor" if not saw_left else "no_usable_right_anchor" if not saw_right
            else "anchor_span_exceeds_limit" if saw_long else "anchor_region_no_exact_reference_match")

class MixedExtractor:
    def __init__(self, forward, reverse):
        self.forward, self.reverse = forward, reverse
        # Shared indices, separate orientation keys: RC collisions cannot overwrite IDs.
        self.lookups = {"forward": forward.lookup, "reverse": reverse.lookup}

    def match(self, sequence):
        first = self.forward.match(sequence)
        if first[0] is not None:
            return first
        second = self.reverse.match(sequence)
        if second[0] is not None:
            return second
        ranks = {"no_left_anchor":0, "no_usable_right_anchor":1, "anchor_span_exceeds_limit":2,
                 "anchor_region_no_exact_reference_match":3, "ambiguous_multiple_guides":4}
        return second if ranks[second[1]]>ranks[first[1]] else first

def extractors(library, left, right, mismatch, policy="legacy", max_extra=32):
    cls = ExactExtractor if mismatch == 0 else TolerantExtractor
    return tuple(cls(library, o, left, right, policy, max_extra) for o in ("forward", "reverse"))

def sniff_orientation(library, path, left=b"CACCG", right=b"GTTTT", anchor_mismatch=0,
                      max_reads=None, policy="legacy", max_extra=32):
    forward, reverse = extractors(library, left, right, anchor_mismatch, policy, max_extra)
    limit = min(20000, max_reads) if max_reads is not None else 20000
    n=f=r=0
    for sequence in fastq_records(path, limit):
        n+=1; f+=forward.match(sequence)[0] is not None; r+=reverse.match(sequence)[0] is not None
    selected = "reverse" if r>f else "forward"
    dominance = max(f,r)/(f+r) if f+r else 0.
    ambiguous = f==r or dominance<.6
    if ambiguous:
        warnings.warn(f"Ambiguous orientation in first {n} records: forward={f}, reverse={r}; "
                      f"using {selected}. Specify orientation or use mixed explicitly.",
                      OrientationWarning, stacklevel=3)
    return selected, {"records":n,"forward_valid_matches":f,"reverse_valid_matches":r,
                      "dominance":dominance,"ambiguous":ambiguous,"selected":selected}

def count_extended(library, path, sample="sample", orientation="forward", left=b"CACCG",
                   right=b"GTTTT", max_reads=None, policy="legacy", max_extra=32, anchor_mismatch=0):
    if isinstance(anchor_mismatch, bool) or type(anchor_mismatch) is not int or anchor_mismatch not in (0,1):
        raise ValueError("anchor_mismatch must be integer 0 or 1")
    if orientation not in ("forward","reverse","auto","mixed"):
        raise ValueError("orientation must be forward, reverse, auto or mixed")
    requested, sniff = orientation, None
    if orientation == "auto":
        orientation, sniff = sniff_orientation(library,path,left,right,anchor_mismatch,max_reads,policy,max_extra)
    if orientation in ("forward","reverse") and anchor_mismatch == 0:
        table, metrics = count_fastq(library,path,sample,orientation,left,right,max_reads,policy,max_extra)
        if sniff is not None:
            metrics.update(orientation_requested=requested,orientation_sniff=sniff)
        return table, metrics
    forward, reverse = extractors(library,left,right,anchor_mismatch,policy,max_extra)
    extractor = MixedExtractor(forward,reverse) if orientation=="mixed" else (forward if orientation=="forward" else reverse)
    counts = np.zeros(len(library.ids),dtype=np.int64)
    categories = Counter()
    wall,cpu=time.perf_counter(),time.process_time()
    for sequence in fastq_records(path,max_reads):
        index,reason=extractor.match(sequence); categories[reason]+=1
        if index is not None: counts[index]+=1
    total,assigned=sum(categories.values()),int(counts.sum())
    if not total: raise ValueError("FASTQ contains no records")
    metrics={"total_reads":total,"mapped_reads":assigned,"mapping_rate_percent":100*assigned/total,
        "categories":dict(categories),"orientation":orientation,"orientation_requested":requested,
        "guide_length":library.length,"anchor_mismatch":anchor_mismatch,
        "left_anchor":(reverse if orientation=="reverse" else forward).left.decode(),
        "right_anchor":(reverse if orientation=="reverse" else forward).right.decode(),
        "algorithm":"exact-guide-opt-in-anchor-orientation","policy":policy,
        "max_extra":max_extra if policy=="unique" else None,"max_reads":max_reads,
        "loop_wall_seconds":time.perf_counter()-wall,"loop_cpu_seconds":time.process_time()-cpu,
        "quality_scores_used":False,"ambiguous_matches_rejected":policy=="unique",
        "mutation_rescue":False,"anchor_substitutions_allowed":anchor_mismatch}
    if sniff is not None: metrics['orientation_sniff']=sniff
    return CountTable(library.ids,library.genes,[sample],counts[:,None]),metrics
