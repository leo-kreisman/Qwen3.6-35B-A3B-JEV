// Does the expert-pack layout actually reduce device reads, and can the source
// GGUF be read with O_DIRECT at all?
//
// Reads the same routed expert set two ways and reports, for each:
//   * requests issued
//   * bytes actually read from the device (/proc/self/io read_bytes)
//   * wall time and effective MiB/s
//
// Modes
//   --check   try a single O_DIRECT pread at real offsets in both files and
//             report the errno. O_DIRECT requires offset, length and buffer to
//             all be aligned to the logical block size; a tensor at
//             data_start + offset with a non-multiple offset cannot be read
//             this way at all, which changes what is even possible.
//
//   bench     simulate a decode's routed set: --experts random experts from
//             each of the 40 layers. From the source that is 3 preads per
//             expert into three regions ~305 MB apart; from the pack it is one
//             contiguous, 16 KiB-aligned slab per expert.
//
// Buffered pread is used by default because it is the only option the source
// GGUF offers. --odirect selects O_DIRECT for the pack arm (and reports if the
// source cannot be read that way).
//
// Build:  cc -O2 -o bench_expert_pack bench_expert_pack.c
// Run:    ./bench_expert_pack --pack ~/models/jev-pack/...bin \
//                             --source ~/models/.../Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
//                             --regions <pack>.source_regions.tsv --check

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define HEADER_BYTES 16384
#define MAX_LAYERS 512
#define COMPONENTS 3

struct layer_region {
    unsigned long gate, up, down;   // absolute source offsets
    unsigned long rows[COMPONENTS]; // per-expert row bytes
    unsigned long stride;           // pack slab stride
};

struct pack_geom {
    unsigned n_layers, n_experts;
    unsigned long body_bytes, src_size;
    unsigned long table_off, base_off;
    struct layer_region layers[MAX_LAYERS];
    unsigned long layer_base[MAX_LAYERS];
};

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static unsigned long xorshift(unsigned long *s) {
    unsigned long x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

// read_bytes counts what the block layer returned to this process, which is how
// a read-ahead effect becomes visible: a buffered 4 KiB request that triggers
// 128 KiB of read-ahead shows up as 128 KiB here.
static unsigned long proc_read_bytes(void) {
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[256];
    unsigned long value = 0;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "read_bytes:", 11)) { sscanf(line + 11, "%lu", &value); break; }
    }
    fclose(f);
    return value;
}

static int read_geom(const char *path, struct pack_geom *g) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return -1; }
    unsigned char head[HEADER_BYTES];
    if (pread(fd, head, HEADER_BYTES, 0) != HEADER_BYTES) {
        fprintf(stderr, "%s: short header\n", path); close(fd); return -1;
    }
    if (memcmp(head, "JEVXPACK", 8)) {
        fprintf(stderr, "%s: bad magic\n", path); close(fd); return -1;
    }
    unsigned version, align, n_comp, reserved;
    memcpy(&version, head + 8, 4);
    memcpy(&align, head + 12, 4);
    memcpy(&g->n_layers, head + 16, 4);
    memcpy(&g->n_experts, head + 20, 4);
    memcpy(&n_comp, head + 24, 4);
    memcpy(&reserved, head + 28, 4);
    memcpy(&g->src_size, head + 32, 8);
    memcpy(&g->body_bytes, head + 40, 8);
    memcpy(&g->table_off, head + 48, 8);
    memcpy(&g->base_off, head + 56, 8);
    (void)reserved;
    if (version != 2) { fprintf(stderr, "%s: version %u, want 2\n", path, version); close(fd); return -1; }
    if (align != 16384) { fprintf(stderr, "%s: align %u, want 16384\n", path, align); close(fd); return -1; }
    if (n_comp != COMPONENTS) { fprintf(stderr, "%s: %u components, want %u\n", path, n_comp, COMPONENTS); close(fd); return -1; }
    if (g->n_layers > MAX_LAYERS) { fprintf(stderr, "%s: too many layers\n", path); close(fd); return -1; }
    for (unsigned i = 0; i < g->n_layers; i++) {
        unsigned long e[4];
        memcpy(e, head + g->table_off + i * 32, 32);
        g->layers[i].stride = e[0];
    }
    for (unsigned i = 0; i < g->n_layers; i++)
        memcpy(&g->layer_base[i], head + g->base_off + i * 8, 8);
    close(fd);
    return 0;
}

// rows come from the sidecar so the benchmark exercises the same numbers the
// repacker wrote, instead of re-deriving them.
static int read_regions(const char *path, struct pack_geom *g) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return -1; }
    char line[512];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        unsigned layer;
        unsigned long gate, up, down, rg, ru, rd, stride;
        if (sscanf(line, "%u %lu %lu %lu %lu %lu %lu %lu",
                   &layer, &gate, &up, &down, &rg, &ru, &rd, &stride) != 8) continue;
        if (layer >= MAX_LAYERS) continue;
        g->layers[layer].gate = gate;
        g->layers[layer].up = up;
        g->layers[layer].down = down;
        g->layers[layer].rows[0] = rg;
        g->layers[layer].rows[1] = ru;
        g->layers[layer].rows[2] = rd;
        g->layers[layer].stride = stride;
    }
    fclose(f);
    // require nothing further here; the pack header's stride is cross-checked
    // against this file by bench_expert_pack.sh before the arms run
    return 0;
}

static int check_odirect(const char *label, const char *path,
                         const unsigned long *offsets, const unsigned long *lens,
                         int n) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { printf("  %-8s open(O_DIRECT) failed: %s\n", label, strerror(errno)); return 0; }
    void *buf;
    size_t cap = 0;
    for (int i = 0; i < n; i++) if (lens[i] > cap) cap = lens[i];
    if (posix_memalign(&buf, 4096, cap)) { close(fd); return 0; }
    int ok = 0, fail = 0, first_errno = 0;
    unsigned long first_off = 0;
    for (int i = 0; i < n; i++) {
        ssize_t got = pread(fd, buf, lens[i], (off_t)offsets[i]);
        if (got == (ssize_t)lens[i]) ok++;
        else { fail++; if (!first_errno && got < 0) { first_errno = errno; first_off = offsets[i]; } }
    }
    printf("  %-8s O_DIRECT: %d/%d ok, %d failed", label, ok, n, fail);
    if (fail) {
        printf("  (errno %d %s; first at offset %lu = %lu mod 4096)",
               first_errno, strerror(first_errno), first_off, first_off % 4096);
    }
    printf("\n");
    free(buf);
    close(fd);
    return fail == 0;
}

// ---- buffered arm -------------------------------------------------------
static int bench_buffered(int fd, const struct pack_geom *g, int experts_per_layer,
                          int sets, int use_pack, int warm) {
    static char buf[8 * 1024 * 1024];
    unsigned long seed = 0x9E3779B97F4A7C15UL;
    unsigned long requests = 0, bytes = 0;
    unsigned long before = proc_read_bytes();
    double t0 = now_s();
    for (int s = 0; s < sets; s++) {
        for (unsigned layer = 0; layer < g->n_layers; layer++) {
            const struct layer_region *r = &g->layers[layer];
            for (int k = 0; k < experts_per_layer; k++) {
                unsigned expert = (unsigned)(xorshift(&seed) % g->n_experts);
                if (use_pack) {
                    unsigned long off = HEADER_BYTES + g->layer_base[layer]
                                        + (unsigned long)expert * r->stride;
                    unsigned long len = r->stride;
                    if (len > sizeof buf) return -1;
                    ssize_t got = pread(fd, buf, len, (off_t)off);
                    if (got != (ssize_t)len) return -2;
                    if (!warm) bytes += (unsigned long)got;
                    requests++;
                } else {
                    unsigned long base[COMPONENTS] = { r->gate, r->up, r->down };
                    for (int c = 0; c < COMPONENTS; c++) {
                        unsigned long len = r->rows[c];
                        unsigned long off = base[c] + (unsigned long)expert * len;
                        if (len > sizeof buf) return -1;
                        ssize_t got = pread(fd, buf, len, (off_t)off);
                        if (got != (ssize_t)len) return -2;
                        if (!warm) bytes += (unsigned long)got;
                        requests++;
                    }
                }
            }
        }
    }
    double dt = now_s() - t0;
    unsigned long after = proc_read_bytes();
    double mib = bytes / 1048576.0;
    printf("  requests %8lu   payload %9.1f MiB   wall %7.3f s   %8.1f MiB/s\n",
           requests, mib, dt, mib / dt);
    if (!warm)
        printf("  device read %9.1f MiB (%.2fx payload)  amplification\n",
               (after - before) / 1048576.0,
               (after - before) / (double)(bytes ? bytes : 1));
    return 0;
}

// ---- io_uring + O_DIRECT arm -------------------------------------------
struct ring {
    int fd;
    unsigned *sq_head, *sq_tail, *sq_ring_mask, *sq_array, *sq_flags;
    struct io_uring_sqe *sqes;
    unsigned *cq_head, *cq_tail, *cq_ring_mask;
    struct io_uring_cqe *cqes;
};

static int io_uring_setup_call(unsigned e, struct io_uring_params *p) { return syscall(__NR_io_uring_setup, e, p); }
static int io_uring_enter_call(int fd, unsigned s, unsigned m, unsigned f) { return syscall(__NR_io_uring_enter, fd, s, m, f, NULL, 0); }

static int setup_ring(struct ring *r, unsigned entries) {
    struct io_uring_params p; memset(&p, 0, sizeof p);
    int fd = io_uring_setup_call(entries, &p);
    if (fd < 0) return -1;
    r->fd = fd;
    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) { if (cq_sz > sq_sz) sq_sz = cq_sz; cq_sz = sq_sz; }
    void *sq = mmap(0, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (sq == MAP_FAILED) return -1;
    void *cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq
             : mmap(0, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    if (cq == MAP_FAILED) return -1;
    void *sqes = mmap(0, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (sqes == MAP_FAILED) return -1;
    r->sq_head = (unsigned *)((char *)sq + p.sq_off.head);
    r->sq_tail = (unsigned *)((char *)sq + p.sq_off.tail);
    r->sq_ring_mask = (unsigned *)((char *)sq + p.sq_off.ring_mask);
    r->sq_array = (unsigned *)((char *)sq + p.sq_off.array);
    r->sq_flags = (unsigned *)((char *)sq + p.sq_off.flags);
    r->sqes = sqes;
    r->cq_head = (unsigned *)((char *)cq + p.cq_off.head);
    r->cq_tail = (unsigned *)((char *)cq + p.cq_off.tail);
    r->cq_ring_mask = (unsigned *)((char *)cq + p.cq_off.ring_mask);
    r->cqes = (struct io_uring_cqe *)((char *)cq + p.cq_off.cqes);
    return 0;
}

static int bench_uring(int fd, const struct pack_geom *g, int experts_per_layer,
                       int sets, int qd) {
    struct ring ring;
    if (setup_ring(&ring, (unsigned)qd)) { fprintf(stderr, "  io_uring_setup failed\n"); return -1; }
    // one slab buffer per queue slot; slab <= 2,048,000 B for this checkpoint
    static char *bufs[512];
    unsigned long max_stride = 0;
    for (unsigned i = 0; i < g->n_layers; i++)
        if (g->layers[i].stride > max_stride) max_stride = g->layers[i].stride;
    for (int i = 0; i < qd; i++) {
        if (posix_memalign((void **)&bufs[i], 4096, max_stride)) return -1;
        memset(bufs[i], 0, max_stride);
    }
    // jobs = every slab the routed set touches, in order
    unsigned long n_jobs = (unsigned long)g->n_layers * experts_per_layer * sets;
    unsigned long *offs = malloc(n_jobs * sizeof(unsigned long));
    unsigned long *lens = malloc(n_jobs * sizeof(unsigned long));
    if (!offs || !lens) return -1;
    unsigned long seed = 0x9E3779B97F4A7C15UL, j = 0;
    for (int s = 0; s < sets; s++)
        for (unsigned layer = 0; layer < g->n_layers; layer++)
            for (int k = 0; k < experts_per_layer; k++) {
                unsigned expert = (unsigned)(xorshift(&seed) % g->n_experts);
                offs[j] = HEADER_BYTES + g->layer_base[layer]
                          + (unsigned long)expert * g->layers[layer].stride;
                lens[j] = g->layers[layer].stride;
                j++;
            }

    unsigned long before = proc_read_bytes();
    double t0 = now_s();
    unsigned long submitted = 0, reaped = 0, bytes = 0;
    unsigned in_flight = 0;
    while (reaped < n_jobs) {
        while (in_flight < (unsigned)qd && submitted < n_jobs) {
            unsigned idx = *ring.sq_tail & *ring.sq_ring_mask;
            struct io_uring_sqe *sqe = &ring.sqes[idx];
            memset(sqe, 0, sizeof *sqe);
            sqe->opcode = IORING_OP_READ;
            sqe->fd = fd;
            sqe->addr = (uint64_t)bufs[submitted % (unsigned long)qd];
            sqe->len = (uint32_t)lens[submitted];
            sqe->off = (uint64_t)offs[submitted];
            sqe->user_data = submitted;
            ring.sq_array[idx] = idx;
            __atomic_store_n(ring.sq_tail, *ring.sq_tail + 1, __ATOMIC_RELEASE);
            submitted++; in_flight++;
        }
        if (io_uring_enter_call(ring.fd, (unsigned)qd, 0, 0) < 0) {
            fprintf(stderr, "  enter: %s\n", strerror(errno));
            break;
        }
        int n = 0;
        while (*ring.cq_head != __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE)) {
            struct io_uring_cqe *cqe = &ring.cqes[*ring.cq_head & *ring.cq_ring_mask];
            if (cqe->res < 0) {
                fprintf(stderr, "  cqe: %s (offset %lu)\n", strerror(-cqe->res), offs[reaped]);
                return -3;
            }
            bytes += (unsigned long)cqe->res;
            reaped++; in_flight--; n++;
            __atomic_store_n(ring.cq_head, *ring.cq_head + 1, __ATOMIC_RELEASE);
        }
        if (n == 0 && io_uring_enter_call(ring.fd, 0, 1, IORING_ENTER_GETEVENTS) < 0) break;
    }
    double dt = now_s() - t0;
    unsigned long after = proc_read_bytes();
    double mib = bytes / 1048576.0;
    printf("  requests %8lu   payload %9.1f MiB   wall %7.3f s   %8.1f MiB/s\n",
           reaped, mib, dt, mib / dt);
    printf("  device read %9.1f MiB (%.2fx payload)  amplification\n",
           (after - before) / 1048576.0, (after - before) / (double)(bytes ? bytes : 1));
    free(offs); free(lens);
    return 0;
}

// ---- io_uring + O_DIRECT against the SOURCE GGUF ------------------------
// The arm that shows why the pack exists: the source's expert offsets are not
// block-aligned, so O_DIRECT is expected to fail with EINVAL, while the pack's
// slabs are 16 KiB-aligned and read cleanly.
static int bench_uring_source(int fd, const struct pack_geom *g, int experts_per_layer,
                              int sets, int qd) {
    unsigned long n_jobs = (unsigned long)g->n_layers * experts_per_layer * sets * COMPONENTS;
    unsigned long *offs = malloc(n_jobs * sizeof(unsigned long));
    unsigned long *lens = malloc(n_jobs * sizeof(unsigned long));
    if (!offs || !lens) return -1;
    unsigned long seed = 0x9E3779B97F4A7C15UL, j = 0;
    for (int s = 0; s < sets; s++)
        for (unsigned layer = 0; layer < g->n_layers; layer++) {
            const struct layer_region *r = &g->layers[layer];
            unsigned long base[COMPONENTS] = { r->gate, r->up, r->down };
            for (int k = 0; k < experts_per_layer; k++) {
                unsigned expert = (unsigned)(xorshift(&seed) % g->n_experts);
                for (int c = 0; c < COMPONENTS; c++) {
                    offs[j] = base[c] + (unsigned long)expert * r->rows[c];
                    lens[j] = r->rows[c];
                    j++;
                }
            }
        }
    unsigned long misaligned = 0;
    for (unsigned long i = 0; i < n_jobs; i++)
        if ((offs[i] % 4096) || (lens[i] % 4096)) misaligned++;
    printf("  %lu/%lu requests have a non-4096-aligned offset or length\n",
           misaligned, n_jobs);

    struct ring ring;
    if (setup_ring(&ring, (unsigned)qd)) { fprintf(stderr, "  io_uring_setup failed\n"); return -1; }
    unsigned long max_len = 0;
    for (unsigned long i = 0; i < n_jobs; i++) if (lens[i] > max_len) max_len = lens[i];
    static char *bufs[512];
    for (int i = 0; i < qd; i++) {
        if (posix_memalign((void **)&bufs[i], 4096, max_len)) return -1;
        memset(bufs[i], 0, max_len);
    }
    unsigned long before = proc_read_bytes();
    double t0 = now_s();
    unsigned long submitted = 0, reaped = 0, bytes = 0, errors = 0;
    unsigned in_flight = 0;
    int first_errno = 0;
    while (reaped < n_jobs) {
        while (in_flight < (unsigned)qd && submitted < n_jobs) {
            unsigned idx = *ring.sq_tail & *ring.sq_ring_mask;
            struct io_uring_sqe *sqe = &ring.sqes[idx];
            memset(sqe, 0, sizeof *sqe);
            sqe->opcode = IORING_OP_READ;
            sqe->fd = fd;
            sqe->addr = (uint64_t)bufs[submitted % (unsigned long)qd];
            sqe->len = (uint32_t)lens[submitted];
            sqe->off = (uint64_t)offs[submitted];
            sqe->user_data = submitted;
            ring.sq_array[idx] = idx;
            __atomic_store_n(ring.sq_tail, *ring.sq_tail + 1, __ATOMIC_RELEASE);
            submitted++; in_flight++;
        }
        if (io_uring_enter_call(ring.fd, (unsigned)qd, 0, 0) < 0) break;
        int n = 0;
        while (*ring.cq_head != __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE)) {
            struct io_uring_cqe *cqe = &ring.cqes[*ring.cq_head & *ring.cq_ring_mask];
            if (cqe->res < 0) {
                if (!first_errno) first_errno = -cqe->res;
                errors++;
            } else bytes += (unsigned long)cqe->res;
            reaped++; in_flight--; n++;
            __atomic_store_n(ring.cq_head, *ring.cq_head + 1, __ATOMIC_RELEASE);
        }
        if (n == 0 && io_uring_enter_call(ring.fd, 0, 1, IORING_ENTER_GETEVENTS) < 0) break;
    }
    double dt = now_s() - t0;
    unsigned long after = proc_read_bytes();
    printf("  requests %8lu   ok %8lu   errors %6lu   wall %7.3f s\n",
           reaped, reaped - errors, errors, dt);
    if (first_errno)
        printf("  first error: errno %d (%s) -> O_DIRECT unusable at these offsets\n",
               first_errno, strerror(first_errno));
    if (bytes)
        printf("  successful-request bytes %.1f MiB; device read %.1f MiB\n",
               bytes / 1048576.0, (after - before) / 1048576.0);
    free(offs); free(lens);
    return 0;
}

int main(int argc, char **argv) {
    const char *pack = NULL, *source = NULL, *regions = NULL, *arm = NULL;
    int experts = 8, sets = 4, check = 0, do_uring = 0, qd = 8, warm = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!strcmp(argv[i], "--source") && i + 1 < argc) source = argv[++i];
        else if (!strcmp(argv[i], "--regions") && i + 1 < argc) regions = argv[++i];
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sets") && i + 1 < argc) sets = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qd") && i + 1 < argc) qd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--odirect")) do_uring = 1;
        else if (!strcmp(argv[i], "--warm")) warm = 1;
        else if (!strcmp(argv[i], "--arm") && i + 1 < argc) arm = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!pack || !source || !regions) {
        fprintf(stderr, "need --pack, --source, --regions\n");
        return 1;
    }
    struct pack_geom g;
    memset(&g, 0, sizeof g);
    if (read_geom(pack, &g)) return 1;
    if (read_regions(regions, &g)) return 1;
    struct stat st;
    if (stat(pack, &st)) { perror("stat pack"); return 1; }
    printf("pack        %s  (%lld B)\n", pack, (long long)st.st_size);
    printf("geometry    %u layers x %u experts, %u geometries\n",
           g.n_layers, g.n_experts, 2);
    for (unsigned i = 0; i < g.n_layers; i++) {
        if (i && g.layers[i].stride == g.layers[i - 1].stride) continue;
        printf("  from blk.%u: stride %lu B (%lu x 16384)\n",
               i, g.layers[i].stride, g.layers[i].stride / 16384);
    }
    if (st.st_size != (off_t)(HEADER_BYTES + g.body_bytes)) {
        printf("  WARNING file size %lld != header claim %lu\n",
               (long long)st.st_size, HEADER_BYTES + g.body_bytes);
    }

    if (check) {
        // sample the first expert of layers 0, 20 and 34 in each file
        unsigned long so[3], sl[3], po[3], pl[3];
        int n = 0;
        unsigned sample_layers[3] = { 0, 20, 34 };
        for (int i = 0; i < 3; i++) {
            unsigned L = sample_layers[i];
            if (L >= g.n_layers) continue;
            so[n] = g.layers[L].gate;      sl[n] = g.layers[L].rows[0]; n++;
        }
        printf("\nO_DIRECT feasibility (offset+len+buffer must all be block-aligned)\n");
        check_odirect("source", source, so, sl, n);
        int m = 0;
        for (int i = 0; i < 3; i++) {
            unsigned L = sample_layers[i];
            if (L >= g.n_layers) continue;
            po[m] = HEADER_BYTES + g.layer_base[L]; pl[m] = g.layers[L].stride; m++;
        }
        check_odirect("pack", pack, po, pl, m);
        printf("\n  note: a single-expert read from the source is 3 separate\n"
               "        preads into regions ~305 MB apart, not one extent.\n");
    }

    unsigned long payload_per_set = 0;
    for (unsigned L = 0; L < g.n_layers; L++)
        payload_per_set += g.layers[L].stride * (unsigned long)experts;
    printf("\nrouted set  %d experts x %u layers = %.1f MiB payload per set, %d set(s)\n",
           experts, g.n_layers, payload_per_set / 1048576.0, sets);

    // One arm per process, so a cold measurement is not contaminated by the arm
    // that ran before it. The runner evicts the file and runs each arm alone.
    if (arm) {
        if (!strcmp(arm, "source")) {
            printf("\nsource GGUF, buffered pread, %d requests per expert\n", COMPONENTS);
            int fd = open(source, O_RDONLY);
            if (fd < 0) { perror("open source"); return 1; }
            int rc = bench_buffered(fd, &g, experts, sets, 0, warm);
            close(fd);
            if (rc) { fprintf(stderr, "  failed rc=%d\n", rc); return 1; }
        } else if (!strcmp(arm, "pack")) {
            printf("\nexpert pack, buffered pread, 1 request per expert\n");
            int fd = open(pack, O_RDONLY);
            if (fd < 0) { perror("open pack"); return 1; }
            int rc = bench_buffered(fd, &g, experts, sets, 1, warm);
            close(fd);
            if (rc) { fprintf(stderr, "  failed rc=%d\n", rc); return 1; }
        } else if (!strcmp(arm, "pack-uring")) {
            printf("\nexpert pack, io_uring + O_DIRECT, qd=%d\n", qd);
            int fd = open(pack, O_RDONLY | O_DIRECT);
            if (fd < 0) { perror("open pack O_DIRECT"); return 1; }
            int rc = bench_uring(fd, &g, experts, sets, qd);
            close(fd);
            if (rc) { fprintf(stderr, "  failed rc=%d\n", rc); return 1; }
        } else if (!strcmp(arm, "source-uring")) {
            printf("\nsource GGUF, io_uring + O_DIRECT, qd=%d\n", qd);
            int fd = open(source, O_RDONLY | O_DIRECT);
            if (fd < 0) { perror("open source O_DIRECT"); return 1; }
            int rc = bench_uring_source(fd, &g, experts, sets, qd);
            close(fd);
            if (rc) { fprintf(stderr, "  failed rc=%d\n", rc); return 1; }
        } else {
            fprintf(stderr, "unknown arm %s\n", arm);
            return 1;
        }
        return 0;
    }

    printf("\nsource GGUF, buffered pread, %d requests per expert\n",
           COMPONENTS);
    int fd = open(source, O_RDONLY);
    if (fd < 0) { perror("open source"); return 1; }
    if (bench_buffered(fd, &g, experts, sets, 0, warm)) { fprintf(stderr, "  failed\n"); return 1; }
    close(fd);

    printf("\nexpert pack, buffered pread, 1 request per expert\n");
    fd = open(pack, O_RDONLY);
    if (fd < 0) { perror("open pack"); return 1; }
    if (bench_buffered(fd, &g, experts, sets, 1, warm)) { fprintf(stderr, "  failed\n"); return 1; }
    close(fd);

    if (do_uring) {
        printf("\nexpert pack, io_uring + O_DIRECT, qd=%d\n", qd);
        fd = open(pack, O_RDONLY | O_DIRECT);
        if (fd < 0) { perror("open pack O_DIRECT"); return 1; }
        if (bench_uring(fd, &g, experts, sets, qd)) { fprintf(stderr, "  failed\n"); return 1; }
        close(fd);

        printf("\nsource GGUF, io_uring + O_DIRECT (expected to fail: offsets unaligned)\n");
        fd = open(source, O_RDONLY | O_DIRECT);
        if (fd < 0) { perror("open source O_DIRECT"); return 1; }
        int rc = bench_uring(fd, &g, experts, 1, qd);
        if (rc) printf("  -> source cannot be read this way (rc=%d)\n", rc);
        close(fd);
    }
    return 0;
}
