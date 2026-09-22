// Raw io_uring (no liburing) vs pread, O_DIRECT random reads on this NVMe.
// Deliberately uses only syscalls + mmap so the same sequence is reproducible
// from Python ctypes.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <errno.h>
#include <linux/io_uring.h>

static int io_uring_setup(unsigned e, struct io_uring_params*p){ return syscall(__NR_io_uring_setup,e,p); }
static int io_uring_enter(int fd,unsigned to_submit,unsigned min_complete,unsigned flags){ return syscall(__NR_io_uring_enter,fd,to_submit,min_complete,flags,NULL,0); }
static int io_uring_register(int fd,unsigned op,void*arg,unsigned nr){ return syscall(__NR_io_uring_register,fd,op,arg,nr); }

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static unsigned long rng_; static unsigned long xr(void){ rng_^=rng_<<13; rng_^=rng_>>7; rng_^=rng_<<17; return rng_; }

struct ring {
  int fd; unsigned *sq_head,*sq_tail,*sq_ring_mask,*sq_ring_entries,*sq_flags,*sq_array;
  struct io_uring_sqe *sqes;
  unsigned *cq_head,*cq_tail,*cq_ring_mask,*cq_entries; struct io_uring_cqe *cqes;
  unsigned sqe_idx;
};

static int setup_ring(struct ring*r, unsigned entries, unsigned flags){
  struct io_uring_params p; memset(&p,0,sizeof p); p.flags=flags;
  int fd = io_uring_setup(entries,&p);
  if (fd<0){ fprintf(stderr,"io_uring_setup: %s\n",strerror(errno)); return -1; }
  r->fd=fd;
  size_t sq_sz = p.sq_off.array + p.sq_entries*sizeof(unsigned);
  size_t cq_sz = p.cq_off.cqes + p.cq_entries*sizeof(struct io_uring_cqe);
  if (p.features & IORING_FEAT_SINGLE_MMAP){ if (cq_sz>sq_sz) sq_sz=cq_sz; cq_sz=sq_sz; }
  void* sq = mmap(0,sq_sz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd,IORING_OFF_SQ_RING);
  if (sq==MAP_FAILED){ fprintf(stderr,"mmap sq: %s\n",strerror(errno)); return -1; }
  void* cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq
             : mmap(0,cq_sz,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_POPULATE,fd,IORING_OFF_CQ_RING);
  if (cq==MAP_FAILED){ fprintf(stderr,"mmap cq: %s\n",strerror(errno)); return -1; }
  void* sqes = mmap(0,p.sq_entries*sizeof(struct io_uring_sqe),PROT_READ|PROT_WRITE,
                    MAP_SHARED|MAP_POPULATE,fd,IORING_OFF_SQES);
  if (sqes==MAP_FAILED){ fprintf(stderr,"mmap sqes: %s\n",strerror(errno)); return -1; }
  r->sq_head=(unsigned*)((char*)sq+p.sq_off.head); r->sq_tail=(unsigned*)((char*)sq+p.sq_off.tail);
  r->sq_ring_mask=(unsigned*)((char*)sq+p.sq_off.ring_mask); r->sq_ring_entries=(unsigned*)((char*)sq+p.sq_off.ring_entries);
  r->sq_flags=(unsigned*)((char*)sq+p.sq_off.flags); r->sq_array=(unsigned*)((char*)sq+p.sq_off.array);
  r->sqes=sqes;
  r->cq_head=(unsigned*)((char*)cq+p.cq_off.head); r->cq_tail=(unsigned*)((char*)cq+p.cq_off.tail);
  r->cq_ring_mask=(unsigned*)((char*)cq+p.cq_off.ring_mask); r->cq_entries=(unsigned*)((char*)cq+p.cq_off.cqes);
  r->cqes=(struct io_uring_cqe*)((char*)cq+p.cq_off.cqes);
  r->sqe_idx=0; return 0;
}

int main(int argc,char**argv){
  if (argc<6){ fprintf(stderr,"usage: %s <file> <bs> <qd> <mode> <MiB> [regbuf]\n",argv[0]); return 1; }
  const char* path=argv[1]; size_t bs=strtoull(argv[2],0,10); int qd=atoi(argv[3]);
  const char* mode=argv[4]; size_t budget=(size_t)strtoull(argv[5],0,10)*1024*1024;
  int use_regbuf = (argc>6);

  void* buf; if (posix_memalign(&buf,4096,(size_t)bs*qd)) return 1;
  memset(buf,0,(size_t)bs*qd);
  struct stat st; if(stat(path,&st)){ perror("stat"); return 1; }
  size_t maxoff = ((size_t)st.st_size - bs) / 4096;
  size_t nreq = budget/bs; if (nreq < (size_t)qd*4) nreq=(size_t)qd*4;
  rng_ = 99991 + bs*7 + qd;

  int fd = open(path,O_RDONLY|O_DIRECT);
  if (fd<0){ perror("open"); return 1; }

  int flags=0;
  if (!strcmp(mode,"iopoll")) flags=IORING_SETUP_IOPOLL;
  else if (!strcmp(mode,"sqpoll")) flags=IORING_SETUP_SQPOLL|IORING_SETUP_SQ_AFF;
  else if (!strcmp(mode,"defer")) flags=IORING_SETUP_DEFER_TASKRUN|IORING_SETUP_SINGLE_ISSUER;

  struct ring r; double t0=now_s(); size_t done=0;
  if (!strcmp(mode,"pread")){
    for (size_t i=0;i<nreq;i++){
      off_t off=(off_t)((xr()%maxoff)*4096);
      ssize_t got=pread(fd,(char*)buf+(i%qd)*bs,bs,off);
      if (got<0){ perror("pread"); return 1; } done+=got;
    }
  } else {
    if (setup_ring(&r,(unsigned)qd,flags)) return 2;
    if (use_regbuf){
      struct iovec iov={.iov_base=buf,.iov_len=(size_t)bs*qd};
      if (io_uring_register(r.fd,IORING_REGISTER_BUFFERS,&iov,1))
        fprintf(stderr,"  [regbuf failed: %s]\n",strerror(errno));
    }
    size_t sub=0,reaped=0; unsigned in_flight=0;
    while (reaped<nreq){
      while (in_flight<(unsigned)qd && sub<nreq){
        unsigned idx = *r.sq_tail & *r.sq_ring_mask;
        struct io_uring_sqe* sqe=&r.sqes[idx];
        memset(sqe,0,sizeof *sqe);
        off_t off=(off_t)((xr()%maxoff)*4096);
        void* b=(char*)buf+(sub%(size_t)qd)*bs;
        sqe->opcode = use_regbuf?IORING_OP_READ_FIXED:IORING_OP_READ;
        sqe->fd=fd; sqe->addr=(uint64_t)b; sqe->len=(uint32_t)bs; sqe->off=(uint64_t)off;
        sqe->user_data=sub;
        if (use_regbuf) sqe->buf_index=0;
        r.sq_array[idx]=idx; __atomic_store_n(r.sq_tail, *r.sq_tail+1, __ATOMIC_RELEASE);
        sub++; in_flight++;
      }
      int ret=io_uring_enter(r.fd,(unsigned)qd,0,0);
      if (ret<0){ fprintf(stderr,"enter: %s\n",strerror(errno)); return 3; }
      int n=0;
      while (*r.cq_head != __atomic_load_n(r.cq_tail,__ATOMIC_ACQUIRE)){
        struct io_uring_cqe* cqe=&r.cqes[*r.cq_head & *r.cq_ring_mask];
        if (cqe->res<0){ fprintf(stderr,"cqe: %s\n",strerror(-cqe->res)); return 4; }
        done+=cqe->res; reaped++; in_flight--; n++;
        __atomic_store_n(r.cq_head,*r.cq_head+1,__ATOMIC_RELEASE);
      }
      if (n==0){ if (io_uring_enter(r.fd,0,1,IORING_ENTER_GETEVENTS)<0){perror("enter wait");return 5;} }
    }
  }
  double dt=now_s()-t0;
  printf("%-30s bs=%-8zu qd=%-4d %8.1f MiB/s  %9.0f IOPS  (%zu MiB / %.3fs)\n",
    use_regbuf?"uring_regbuf":"uring",bs,qd,(done/1048576.0)/dt, done/bs/dt, done/1048576, dt);
  if (!strcmp(mode,"pread")) { /* label */ }
  free(buf); close(fd); return 0;
}
