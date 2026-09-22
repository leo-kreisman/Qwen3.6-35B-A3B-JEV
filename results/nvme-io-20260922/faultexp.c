// How many bytes does ONE page fault actually pull into the page cache from
// the GGUF mapping, and do MADV_RANDOM / MADV_HUGEPAGE / MADV_COLLAPSE change it?
// This is the mechanism behind the project's measured ~2.15x read amplification.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

static long resident_bytes(void* addr, size_t len){
  long ps = sysconf(_SC_PAGESIZE);
  unsigned char vec[1<<20];
  size_t npages = len/ps; if (npages > (1<<20)) npages = (1<<20);
  if (mincore(addr, npages*ps, vec)) { perror("mincore"); return -1; }
  long n=0; for (size_t i=0;i<npages;i++) if (vec[i]&1) n++;
  return n*ps;
}

int main(int argc,char**argv){
  const char* path = argv[1];
  int mode = atoi(argv[2]); // 0=none 1=MADV_RANDOM 2=MADV_HUGEPAGE 3=MADV_COLLAPSE(after normal) 4=MADV_SEQUENTIAL
  size_t win = (size_t)strtoull(argv[3],0,10)<<20;  // window to sample
  size_t step = strtoull(argv[4],0,10);             // stride between probes

  int fd = open(path, O_RDONLY);
  if (fd<0){ perror("open"); return 1; }
  struct stat st; fstat(fd,&st);
  // Evict the sampled window from page cache BEFORE mapping it — page-cache
  // pages that are pinned by an active VMA will not be dropped by fadvise.
  if (posix_fadvise(fd, 0, win, POSIX_FADV_DONTNEED)) perror("fadvise");
  // llama.cpp maps the model read-only; PROT_READ|PROT_EXEC is required for
  // file-backed THP, so test both.
  int exec_map = (mode==2||mode==3);
  void* addr = mmap(NULL, win, PROT_READ|(exec_map?PROT_EXEC:0),
                    MAP_SHARED|MAP_NORESERVE, fd, 0);
  if (addr==MAP_FAILED){ printf("mmap failed: %s\n",strerror(errno)); return 1; }

  const char* label="none";
  if (mode==1){ madvise(addr,win,MADV_RANDOM); label="MADV_RANDOM"; }
  else if (mode==2){ madvise(addr,win,MADV_HUGEPAGE); label="MADV_HUGEPAGE"; }
  else if (mode==4){ madvise(addr,win,MADV_SEQUENTIAL); label="MADV_SEQUENTIAL"; }

  long before = resident_bytes(addr,win);
  volatile unsigned char sink=0;
  long total_fetched=0; int faults=0;
  for (size_t off=0; off+4096<win; off+=step){
    long r0 = resident_bytes(addr,win);
    sink += ((volatile unsigned char*)addr)[off];
    long r1 = resident_bytes(addr,win);
    long delta = r1-r0;
    if (delta>0){ total_fetched += delta; faults++; }
  }
  int rc=0;
  if (mode==3){ rc = madvise(addr,win,25 /*MADV_COLLAPSE since 6.1*/); label = rc==0?"MADV_COLLAPSE ok":"MADV_COLLAPSE FAILED"; }
  long after = resident_bytes(addr,win);

  printf("mode=%-18s win=%zuMiB step=%zuB faults=%-6d fetched=%8.1f MiB  "
         "avg_bytes_per_fault=%8ld  resident: before=%.1f after=%.1f MiB%s\n",
         label, win>>20, step, faults, total_fetched/1048576.0, faults? total_fetched/faults : 0,
         before/1048576.0, after/1048576.0, (mode==3&&rc)?"  (collapse errno=EINVAL/ENOMEM)":"");
  (void)sink;
  munmap(addr,win); close(fd);
  return 0;
}
