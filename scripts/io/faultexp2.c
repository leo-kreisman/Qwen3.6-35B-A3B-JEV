// Extension of faultexp.c by the same author: adds the MADV_POPULATE_READ variant
// that faultexp.c did not implement, plus a whole-slice populate that models the
// real expert gather (one 576 KiB slice at a time).
//   mode 5 = MADV_POPULATE_READ on the single 4 KiB page at each stride offset
//   mode 6 = MADV_POPULATE_READ on the whole 589824 B slice at each stride offset
//   mode 7 = POSIX_FADV_RANDOM on the fd + ordinary page fault (no madvise)
// Everything else (eviction before mmap, mincore accounting, stride) is identical
// to faultexp.c so the numbers are directly comparable.
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
  static unsigned char vec[1<<20];
  size_t npages = len/ps; if (npages > (1<<20)) npages = (1<<20);
  if (mincore(addr, npages*ps, vec)) { perror("mincore"); return -1; }
  long n=0; for (size_t i=0;i<npages;i++) if (vec[i]&1) n++;
  return n*ps;
}

int main(int argc,char**argv){
  const char* path = argv[1];
  int mode = atoi(argv[2]); // 5=populate page, 6=populate slice, 7=fadvise RANDOM
  size_t win = (size_t)strtoull(argv[3],0,10)<<20;
  size_t step = strtoull(argv[4],0,10);

  int fd = open(path, O_RDONLY);
  if (fd<0){ perror("open"); return 1; }
  struct stat st; fstat(fd,&st);
  if (posix_fadvise(fd, 0, win, POSIX_FADV_DONTNEED)) perror("fadvise");
  if (mode==7 && posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM))
    perror("fadvise RANDOM");
  if (mode==8 && posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL))
    perror("fadvise SEQUENTIAL");   // sets file->f_ra.ra_pages = bdi->ra_pages*2

  void* addr = mmap(NULL, win, PROT_READ, MAP_SHARED|MAP_NORESERVE, fd, 0);
  if (addr==MAP_FAILED){ printf("mmap failed: %s\n",strerror(errno)); return 1; }

  long before = resident_bytes(addr,win);
  volatile unsigned char sink=0;
  long total_fetched=0; int faults=0; int nerr=0;
  for (size_t off=0; off+4096<win; off+=step){
    long r0 = resident_bytes(addr,win);
    if (mode==5){
      if (madvise((char*)addr+off, 4096, MADV_POPULATE_READ)) nerr++;
    } else if (mode==6){
      size_t len = step;                       // one whole expert slice
      if (off+len>win) len = win-off;
      if (madvise((char*)addr+off, len, MADV_POPULATE_READ)) nerr++;
    } else {
      sink += ((volatile unsigned char*)addr)[off];
    }
    long r1 = resident_bytes(addr,win);
    long delta = r1-r0;
    if (delta>0){ total_fetched += delta; faults++; }
  }
  long after = resident_bytes(addr,win);
  const char* label = mode==5?"MADV_POPULATE_READ(4K)":mode==6?"MADV_POPULATE_READ(slice)":mode==7?"FADV_RANDOM+fault":mode==8?"FADV_SEQUENTIAL+fault":"?";
  printf("mode=%-24s win=%zuMiB step=%zuB faults=%-6d fetched=%8.1f MiB  "
         "avg_bytes_per_fault=%8ld  resident: before=%.1f after=%.1f MiB  errs=%d\n",
         label, win>>20, step, faults, total_fetched/1048576.0,
         faults? total_fetched/faults : 0, before/1048576.0, after/1048576.0, nerr);
  (void)sink;
  munmap(addr,win); close(fd);
  return 0;
}
