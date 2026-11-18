import json
import resource
import sys
import time
from pathlib import Path

mode,fastq,reference,n,output=sys.argv[1:]
n=int(n); started=time.perf_counter(); cpu=time.process_time()
if mode=='baseline':
    from readsparrow.counting import count_fastq,load_library
    table,metrics=count_fastq(load_library(reference),fastq,'count','forward',max_reads=n)
    counts=table.counts[:,0].tolist()
elif mode=='strict':
    from readsparrow.counting import load_library
    from readsparrow.orientation import count_extended
    table,metrics=count_extended(load_library(reference),fastq,'count','forward',max_reads=n)
    counts=table.counts[:,0].tolist()
else:
    import readsparrow as rs
    orientation={'api':'forward','auto':'auto','tolerant':'forward','mixed':'mixed'}[mode]
    result=rs.count(fastq,reference,orientation=orientation,anchor_mismatch=int(mode=='tolerant'),max_reads=n)
    counts=result.counts_df['count'].tolist(); metrics=result.qc_summary
usage=resource.getrusage(resource.RUSAGE_SELF)
record={'mode':mode,'records':n,'counts':counts,'metrics':metrics,'CPU_seconds':time.process_time()-cpu,
        'elapsed_inside_seconds':time.perf_counter()-started,'user_seconds':usage.ru_utime,
        'system_seconds':usage.ru_stime,'peak_RSS_bytes':usage.ru_maxrss if sys.platform=='darwin' else usage.ru_maxrss*1024,
        'python':sys.version,'platform':sys.platform}
Path(output).write_text(json.dumps(record,indent=2))
