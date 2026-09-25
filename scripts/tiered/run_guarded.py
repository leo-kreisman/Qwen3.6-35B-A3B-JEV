#!/usr/bin/env python3
"""Run INSIDE an 8 GiB systemd scope; record peak, events and process-tree reads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def io_bytes():
    return int(dict(line.split(':',1) for line in Path('/proc/self/io').read_text().splitlines())['read_bytes'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--evict',type=Path,action='append',default=[])
    ap.add_argument('command',nargs=argparse.REMAINDER)
    a=ap.parse_args(); cmd=a.command
    if cmd and cmd[0]=='--': cmd=cmd[1:]
    if not cmd: ap.error('missing command')
    group=next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    cg=Path('/sys/fs/cgroup')/group.lstrip('/')
    limit=(cg/'memory.max').read_text().strip()
    if limit=='max' or int(limit)>8*1024**3:
        raise RuntimeError('run inside systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0')
    if (cg/'memory.swap.max').read_text().strip()!='0': raise RuntimeError('swap must be disabled')
    if a.output.exists() or a.output.with_suffix('.log').exists(): raise FileExistsError(a.output)
    cache=[]
    for path in a.evict:
        fd=os.open(path,os.O_RDONLY)
        try: os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally: os.close(fd)
        cached=int(subprocess.check_output(['fincore','--bytes','--noheadings','--output','RES',str(path)],text=True).strip())
        cache.append(dict(path=str(path),cached_bytes=cached))
        if cached: raise RuntimeError(f'cannot establish cold cache: {path}: {cached} bytes')
    before=io_bytes(); started=time.monotonic()
    with a.output.with_suffix('.log').open('x') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    report=dict(command=cmd,wall_seconds=time.monotonic()-started,exit_code=r.returncode,
                process_tree_read_bytes=io_bytes()-before,memory_max=int(limit),
                memory_peak=int((cg/'memory.peak').read_text()),
                memory_events={k:int(v) for k,v in (line.split() for line in (cg/'memory.events').read_text().splitlines())},
                cold_cache=cache)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    raise SystemExit(r.returncode)

if __name__=='__main__': main()
