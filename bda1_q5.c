/*
 * BDA Assignment-1 Q5: Central Tendency of Streamed Data — Log-Normal Distribution
 *
 * Simulates 100,000 values/second for 1 hour (360,000,000 total values).
 * Distribution: Log-Normal(mu=0, sigma=0.5) clipped to [0, 1].
 *   Median of raw = e^0 = 1.0; after clipping most mass is right-skewed in [0,1].
 *   Mean of clipped ≈ 0.60-0.65 (right of centre, as expected for log-normal).
 *
 * Outputs:
 *   - Overall stats (Part a)
 *   - Per-minute stats to minute_stats_ln.csv (Part b)
 *   - 10-minute block stats to ten_min_stats_ln.csv (Part b)
 *   - IQR / outlier analysis (Part c)
 *   - global_stats_ln.csv for box-plot script
 *
 * Compile (Linux/macOS):
 *   gcc -O2 -o q5_lognormal q5_lognormal.c -lm -lpthread
 *
 * Compile (Windows with MinGW):
 *   gcc -O2 -o q5_lognormal q5_lognormal.c -lm
 *   (Windows path uses CreateThread automatically via the #ifdef below)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
  #include <windows.h>
  typedef HANDLE thread_t;
  #define THREAD_FUNC DWORD WINAPI
  static void thread_create(thread_t *h, LPTHREAD_START_ROUTINE fn, void *arg)
      { *h = CreateThread(NULL, 0, fn, arg, 0, NULL); }
  static void thread_join(thread_t h)
      { WaitForSingleObject(h, INFINITE); CloseHandle(h); }
  #include <intrin.h>
  #define ATOMIC_ADD(ptr, val) _InterlockedExchangeAdd((volatile LONG *)(ptr), (val))
#else
  #include <pthread.h>
  typedef pthread_t thread_t;
  #define THREAD_FUNC void *
  static void thread_create(thread_t *h, void *(*fn)(void *), void *arg)
      { pthread_create(h, NULL, fn, arg); }
  static void thread_join(thread_t h) { pthread_join(h, NULL); }
  #define ATOMIC_ADD(ptr, val) __sync_fetch_and_add((ptr), (val))
#endif

/* ── Simulation parameters ───────────────────────────────────────────── */
#define THREADS          4
#define BINS          2000          /* histogram resolution                */
#define FIXED_SEED      42U
#define TOTAL_SECONDS 3600          /* 1 hour                              */
#define VALUES_PER_SEC 100000L

/* Log-Normal parameters (underlying normal) */
#define LN_MU    0.0
#define LN_SIGMA 0.5

#define TOTAL_MINUTES  (TOTAL_SECONDS / 60)    /* 60  */
#define TEN_MIN_BLOCKS (TOTAL_SECONDS / 600)   /* 6   */
#define VALUES_PER_MIN (VALUES_PER_SEC * 60L)
#define VALUES_PER_10M (VALUES_PER_SEC * 600L)
#define TOTAL_VALUES   ((long)VALUES_PER_SEC * TOTAL_SECONDS)

/* ── Histogram storage ───────────────────────────────────────────────── */
static long  global_hist[BINS];
static long *minute_hist;   /* [TOTAL_MINUTES][BINS]  */
static long *ten_min_hist;  /* [TEN_MIN_BLOCKS][BINS] */

#define MIN_HIST(m, b)  minute_hist [(m)*BINS + (b)]
#define TEN_HIST(t, b)  ten_min_hist[(t)*BINS + (b)]

/* ── LCG PRNG ────────────────────────────────────────────────────────── */
static inline unsigned int lcg_next(unsigned int *s)
{
    *s = *s * 1664525u + 1013904223u;
    return *s;
}
static inline double lcg_uniform(unsigned int *s)
{
    return (lcg_next(s) & 0x7FFFFFFFu) / (double)0x7FFFFFFFu;
}

/* ── Log-Normal sample clipped to [0,1] ─────────────────────────────── */
static double lognormal_sample(unsigned int *s)
{
    /* Box-Muller (Marsaglia polar variant for speed) */
    static __thread int    have_spare = 0;
    static __thread double spare      = 0.0;

    double z;
    if (have_spare) {
        z = spare; have_spare = 0;
    } else {
        double u, v, r;
        do {
            u = 2.0 * lcg_uniform(s) - 1.0;
            v = 2.0 * lcg_uniform(s) - 1.0;
            r = u*u + v*v;
        } while (r >= 1.0 || r == 0.0);
        double fac = sqrt(-2.0 * log(r) / r);
        spare      = v * fac;
        have_spare = 1;
        z          = u * fac;
    }

    /* Log-Normal: X = e^(mu + sigma*Z) */
    double val = exp(LN_MU + LN_SIGMA * z);

    /* Clip to [0, 1] — raw values > 1 are clamped */
    if (val < 0.0) val = 0.0;
    if (val > 1.0) val = 1.0;
    return val;
}

/* ── Worker thread ───────────────────────────────────────────────────── */
typedef struct { int thread_id; int start_sec; int end_sec; } ThreadData;

THREAD_FUNC worker(void *arg)
{
    ThreadData   *t    = (ThreadData *)arg;
    unsigned int  seed = FIXED_SEED + (unsigned int)t->thread_id * 2654435761u;

    for (int sec = t->start_sec; sec < t->end_sec; sec++) {
        int minute  = sec / 60;
        int ten_min = sec / 600;

        for (long i = 0; i < VALUES_PER_SEC; i++) {
            double val = lognormal_sample(&seed);
            int bin = (int)(val * BINS);
            if (bin >= BINS) bin = BINS - 1;

            ATOMIC_ADD(&global_hist[bin],          1L);
            ATOMIC_ADD(&MIN_HIST(minute,  bin),    1L);
            ATOMIC_ADD(&TEN_HIST(ten_min, bin),    1L);
        }
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

/* ── Statistics from a histogram ─────────────────────────────────────── */
static void compute_stats(long *hist, long total,
                           double *mean,  double *min_v, double *max_v,
                           double *median, double *p25,  double *p75,
                           double *mode,  double *stddev)
{
    long   cumul   = 0;
    double sum     = 0.0, sum_sq = 0.0;
    long   max_cnt = 0;
    int    mode_bin = 0;

    *min_v = -1.0; *max_v = -1.0;
    *median = *p25 = *p75 = 0.0;

    for (int i = 0; i < BINS; i++) {
        if (!hist[i]) continue;
        double bc = (i + 0.5) / (double)BINS;
        if (*min_v < 0) *min_v = bc;
        *max_v  = bc;
        sum    += hist[i] * bc;
        sum_sq += hist[i] * bc * bc;
        if (hist[i] > max_cnt) { max_cnt = hist[i]; mode_bin = i; }
    }

    *mean   = sum    / (double)total;
    *mode   = (mode_bin + 0.5) / (double)BINS;
    *stddev = sqrt(sum_sq / (double)total - (*mean) * (*mean));

    long q25 = (long)(total * 0.25);
    long q50 = (long)(total * 0.50);
    long q75 = (long)(total * 0.75);

    for (int i = 0; i < BINS; i++) {
        cumul += hist[i];
        double bc = (i + 0.5) / (double)BINS;
        if (*p25    == 0.0 && cumul >= q25) *p25    = bc;
        if (*median == 0.0 && cumul >= q50) *median = bc;
        if (*p75    == 0.0 && cumul >= q75) *p75    = bc;
    }
}

static void print_stats(const char *label,
                         double mean,  double min_v, double max_v,
                         double median, double p25,  double p75,
                         double mode,  double stddev)
{
    printf("%-22s Mean=%7.5f  StdDev=%7.5f  Min=%7.5f  Max=%7.5f  "
           "Median=%7.5f  P25=%7.5f  P75=%7.5f  Mode=%7.5f\n",
           label, mean, stddev, min_v, max_v, median, p25, p75, mode);
}

/* ═══════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  Log-Normal Stream Statistics Simulator (mu=0, sigma=0.5)  ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");
    printf("  Duration       : %d s  (1 hour)\n",  TOTAL_SECONDS);
    printf("  Values/second  : %ld\n",              VALUES_PER_SEC);
    printf("  Total values   : %ld\n",              TOTAL_VALUES);
    printf("  Minute windows : %d\n",               TOTAL_MINUTES);
    printf("  10-min blocks  : %d\n",               TEN_MIN_BLOCKS);
    printf("  Worker threads : %d\n",               THREADS);
    printf("  Fixed seed     : %u\n\n",             FIXED_SEED);
    printf("  Theoretical (Log-Normal clipped to [0,1]):\n");
    printf("    Raw median = e^0 = 1.0 (at boundary, so clipped median < 1)\n");
    printf("    Raw mean   = e^(sigma^2/2) = e^0.125 ≈ 1.133 (clipped to ~0.6)\n");
    printf("    Distribution is RIGHT-SKEWED: mean > median > mode\n\n");

    memset(global_hist, 0, sizeof(global_hist));
    minute_hist  = calloc((size_t)TOTAL_MINUTES  * BINS, sizeof(long));
    ten_min_hist = calloc((size_t)TEN_MIN_BLOCKS * BINS, sizeof(long));
    if (!minute_hist || !ten_min_hist) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    printf("  Running simulation (Log-Normal via Box-Muller)...");
    fflush(stdout);

    thread_t   threads[THREADS];
    ThreadData tdata[THREADS];
    int chunk = TOTAL_SECONDS / THREADS;

    for (int i = 0; i < THREADS; i++) {
        tdata[i].thread_id = i;
        tdata[i].start_sec = i * chunk;
        tdata[i].end_sec   = (i == THREADS - 1) ? TOTAL_SECONDS : (i + 1) * chunk;
        thread_create(&threads[i], worker, &tdata[i]);
    }
    for (int i = 0; i < THREADS; i++)
        thread_join(threads[i]);

    printf(" done.\n\n");

    double mean, min_v, max_v, median, p25, p75, mode, stddev;

    /* ── Part (a): Overall statistics ─────────────────────────────────── */
    compute_stats(global_hist, TOTAL_VALUES,
                  &mean, &min_v, &max_v, &median, &p25, &p75, &mode, &stddev);

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║           PART (a): OVERALL STATISTICS               ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    print_stats("Overall:", mean, min_v, max_v, median, p25, p75, mode, stddev);

    printf("\n  Significance of measures for Log-Normal:\n");
    printf("    Mean   : Pulled right by the long tail; > median for log-normal.\n");
    printf("    Median : More representative centre; = e^mu before clipping.\n");
    printf("    Mode   : Peak of the skewed bell; < median < mean (right-skew).\n");
    printf("    StdDev : Reflects spread; larger relative to mean than Gaussian.\n");
    printf("    P25/P75: IQR captures the bulk; IQR is narrower on the left tail.\n");
    printf("    Min/Max: Empirical bounds; clipping at 1 creates a pile-up at upper fence.\n");
    printf("    Actual got -> Mean=%.5f Median=%.5f Mode=%.5f\n", mean, median, mode);

    /* ── Part (b): Per-minute analysis ───────────────────────────────── */
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║         PART (b): PER-MINUTE ANALYSIS                ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    printf("%-22s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
           "Interval","Mean","StdDev","Min","Max","Median","P25","P75","Mode");

    FILE *csv_min = fopen("minute_stats_ln.csv", "w");
    fprintf(csv_min, "minute,mean,stddev,min,max,median,p25,p75,mode\n");

    /* Anomaly: flag if |mean - overall_mean| > 3 * stddev/sqrt(N_per_min) */
    double expected_se = stddev / sqrt((double)VALUES_PER_MIN);
    int anomaly_found  = 0;

    for (int m = 0; m < TOTAL_MINUTES; m++) {
        double mn, mi, ma, med, q1, q3, mo, sd;
        compute_stats(&MIN_HIST(m, 0), VALUES_PER_MIN,
                      &mn, &mi, &ma, &med, &q1, &q3, &mo, &sd);

        char label[32];
        snprintf(label, sizeof(label), "Minute %02d:", m + 1);
        print_stats(label, mn, mi, ma, med, q1, q3, mo, sd);

        if (fabs(mn - mean) > 3.0 * expected_se) {
            printf("  *** ANOMALY in Minute %02d: mean=%.5f deviates >3 SE ***\n",
                   m + 1, mn);
            anomaly_found = 1;
        }

        fprintf(csv_min, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                m + 1, mn, sd, mi, ma, med, q1, q3, mo);
    }
    fclose(csv_min);

    if (!anomaly_found)
        printf("\n  No anomalies detected (all minute means within 3 SE of overall mean).\n");

    printf("\n  Trend interpretation:\n");
    printf("    Log-Normal values are i.i.d.; per-minute means should hover at ~%.5f\n", mean);
    printf("    with SE ≈ %.7f. Right-skew means occasional high-value spikes\n", expected_se);
    printf("    can push a minute mean above the typical range (detectable anomalies).\n");

    /* 10-minute blocks */
    printf("\n╔════════════════════════════════════════════════════╗\n");
    printf("║         10-MINUTE BLOCK ANALYSIS                   ║\n");
    printf("╚════════════════════════════════════════════════════╝\n");
    printf("%-22s %-9s %-9s %-9s %-9s %-9s %-9s %-9s %-9s\n",
           "Interval","Mean","StdDev","Min","Max","Median","P25","P75","Mode");

    FILE *csv_10 = fopen("ten_min_stats_ln.csv", "w");
    fprintf(csv_10, "block,mean,stddev,min,max,median,p25,p75,mode\n");

    for (int b = 0; b < TEN_MIN_BLOCKS; b++) {
        double mn, mi, ma, med, q1, q3, mo, sd;
        compute_stats(&TEN_HIST(b, 0), VALUES_PER_10M,
                      &mn, &mi, &ma, &med, &q1, &q3, &mo, &sd);

        char label[32];
        snprintf(label, sizeof(label), "Min %02d-%02d:", b*10+1, (b+1)*10);
        print_stats(label, mn, mi, ma, med, q1, q3, mo, sd);

        fprintf(csv_10, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                b + 1, mn, sd, mi, ma, med, q1, q3, mo);
    }
    fclose(csv_10);

    /* ── Part (c): IQR & Outlier analysis ────────────────────────────── */
    /* Recompute global stats (already in variables above) */
    double IQR   = p75 - p25;
    double lower = p25 - 1.5 * IQR;
    double upper = p75 + 1.5 * IQR;

    long outlier_count = 0;
    for (int i = 0; i < BINS; i++) {
        double bc = (i + 0.5) / (double)BINS;
        if (bc < lower || bc > upper)
            outlier_count += global_hist[i];
    }

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║       PART (c): IQR & OUTLIER ANALYSIS               ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    printf("  Q1 (P25)     : %.5f\n", p25);
    printf("  Q3 (P75)     : %.5f\n", p75);
    printf("  IQR          : %.5f\n", IQR);
    printf("  Lower fence  : %.5f   (Q1 - 1.5 × IQR)\n", lower);
    printf("  Upper fence  : %.5f   (Q3 + 1.5 × IQR)\n", upper);
    printf("  Outlier range: values < %.5f  OR  > %.5f\n", lower, upper);
    printf("  Outlier count: %ld  (%.4f%% of %ld values)\n",
           outlier_count,
           100.0 * outlier_count / (double)TOTAL_VALUES,
           TOTAL_VALUES);
    printf("\n  Log-Normal has a HEAVIER right tail than Gaussian.\n");
    printf("  Expect more upper outliers (>> 0.7%% of Gaussian).\n");
    printf("  Clipping at 1.0 concentrates mass at the upper boundary,\n");
    printf("  so the upper fence outlier count is inflated.\n");
    printf("\n  Impact of outliers on central tendency (Log-Normal):\n");
    printf("    Mean   : Significantly pulled upward by right-tail outliers.\n");
    printf("    Median : Robust — barely affected by upper tail.\n");
    printf("    Mode   : Unaffected — peak is well below the tail.\n");
    printf("    StdDev : Inflated by large upper-tail values.\n");

    /* Save CSV for box-plot script */
    FILE *csv_g = fopen("global_stats_ln.csv", "w");
    fprintf(csv_g, "interval,mean,stddev,min,max,median,p25,p75,mode\n");
    fprintf(csv_g, "Overall,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            mean, stddev, min_v, max_v, median, p25, p75, mode);
    for (int b = 0; b < TEN_MIN_BLOCKS; b++) {
        double mn, mi, ma, med, q1, q3, mo, sd;
        compute_stats(&TEN_HIST(b, 0), VALUES_PER_10M,
                      &mn, &mi, &ma, &med, &q1, &q3, &mo, &sd);
        fprintf(csv_g, "Block_%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                b + 1, mn, sd, mi, ma, med, q1, q3, mo);
    }
    fclose(csv_g);

    printf("\n  CSV files saved:\n");
    printf("    global_stats_ln.csv   — overall + 10-min blocks\n");
    printf("    minute_stats_ln.csv   — per-minute stats\n");
    printf("    ten_min_stats_ln.csv  — 10-minute block stats\n");
    printf("  Run plot_boxplots_ln.py to generate box plots.\n\n");

    free(minute_hist);
    free(ten_min_hist);
    return 0;
}
