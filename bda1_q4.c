/*
 * BDA Assignment-1 Q4: Median-of-Medians — Log-Normal Distribution
 * Total elements : 10^9
 * Range          : [-2^30, 2^30]
 *
 * Strategy (same histogram approach as Gaussian version):
 *   Generate X ~ LogNormal(mu, sigma) shifted & scaled to fill [-2^30, 2^30].
 *   Use a coarse histogram to locate the median bucket, then
 *   apply the median-of-local-medians (median-of-medians) estimate.
 *
 * Compile: gcc -O2 -o q4_lognormal q4_lognormal.c -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#define TOTAL_ELEMENTS 1000000000ULL
#define NUM_THREADS    8
#define NUM_BUCKETS    1024

#define RANGE_MIN  (-1073741824LL)   /* -2^30 */
#define RANGE_MAX  ( 1073741824LL)   /*  2^30 */
#define RANGE_SPAN  2147483649ULL    /* RANGE_MAX - RANGE_MIN + 1 */

/*
 * Log-Normal parameterisation shifted to cover [-2^30, 2^30]:
 *   We generate U ~ LogNormal(0, 0.8), which is positive.
 *   Then map: val = RANGE_MIN + U / E[U] * RANGE_SPAN/2
 *   so the median lands near the centre.
 *   E[X] for LogNormal(0,0.8) = e^(0 + 0.32) ≈ 1.377
 */
#define LN_MU    0.0
#define LN_SIGMA 0.8
/* scale factor: maps median (=1.0) to mid-range */
#define LN_SCALE ((double)RANGE_SPAN / 2.0)

typedef struct {
    unsigned long long start;
    unsigned long long end;
    unsigned long long bucket_counts[NUM_BUCKETS];
} ThreadArgs;

/* Box-Muller */
static double std_normal_r(unsigned int *seed)
{
    double u1, u2;
    do { u1 = (double)rand_r(seed) / RAND_MAX; } while (u1 < 1e-10);
    u2 = (double)rand_r(seed) / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static inline int get_bucket(long long val)
{
    unsigned long long shifted = (unsigned long long)(val - RANGE_MIN);
    int b = (int)(shifted * NUM_BUCKETS / RANGE_SPAN);
    if (b < 0)           b = 0;
    if (b >= NUM_BUCKETS) b = NUM_BUCKETS - 1;
    return b;
}

void *histogram_worker(void *arg)
{
    ThreadArgs *a = (ThreadArgs *)arg;
    memset(a->bucket_counts, 0, sizeof(a->bucket_counts));

    unsigned int seed = (unsigned int)(a->start + 456789);

    unsigned long long count = a->end - a->start;
    for (unsigned long long i = 0; i < count; i++) {
        double z   = std_normal_r(&seed);
        double raw = exp(LN_MU + LN_SIGMA * z); /* raw > 0 */
        /* shift: centre raw median (=1.0) at RANGE_MIN + RANGE_SPAN/2 = 0 */
        double val_d = RANGE_MIN + raw * LN_SCALE;

        /* clamp */
        if (val_d < RANGE_MIN) val_d = RANGE_MIN;
        if (val_d > RANGE_MAX) val_d = RANGE_MAX;

        long long val = (long long)val_d;
        a->bucket_counts[get_bucket(val)]++;
    }

    return NULL;
}

int main(void)
{
    printf("BDA Assignment-1 Q4: Median-of-Medians (Log-Normal)\n");
    printf("Total elements : 10^9 = %llu\n", TOTAL_ELEMENTS);
    printf("Range          : -2^30 to 2^30\n");
    printf("Distribution   : Log-Normal(mu=%.1f, sigma=%.1f) shifted to range\n",
           LN_MU, LN_SIGMA);
    printf("Threads        : %d\n\n", NUM_THREADS);

    pthread_t  threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    unsigned long long chunk = TOTAL_ELEMENTS / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        args[t].start = (unsigned long long)t * chunk;
        args[t].end   = (t == NUM_THREADS - 1) ? TOTAL_ELEMENTS
                                                : (unsigned long long)(t + 1) * chunk;
        pthread_create(&threads[t], NULL, histogram_worker, &args[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    /* Merge histograms */
    unsigned long long global_hist[NUM_BUCKETS];
    memset(global_hist, 0, sizeof(global_hist));
    for (int t = 0; t < NUM_THREADS; t++)
        for (int b = 0; b < NUM_BUCKETS; b++)
            global_hist[b] += args[t].bucket_counts[b];

    /* Global median via histogram */
    unsigned long long median_rank = TOTAL_ELEMENTS / 2;
    unsigned long long cumsum      = 0;
    int median_bucket              = 0;
    for (int b = 0; b < NUM_BUCKETS; b++) {
        cumsum += global_hist[b];
        if (cumsum >= median_rank) { median_bucket = b; break; }
    }

    long long bucket_lo = RANGE_MIN +
        (long long)((unsigned long long)median_bucket       * RANGE_SPAN / NUM_BUCKETS);
    long long bucket_hi = RANGE_MIN +
        (long long)((unsigned long long)(median_bucket + 1) * RANGE_SPAN / NUM_BUCKETS);
    long long median_estimate = bucket_lo + (bucket_hi - bucket_lo) / 2;

    printf("Median bucket   : %d\n",   median_bucket);
    printf("Bucket range    : [%lld, %lld]\n", bucket_lo, bucket_hi);
    printf("Median estimate : %lld\n\n", median_estimate);

    /* Local medians per thread (median-of-medians) */
    printf("Local median estimates per thread:\n");
    long long local_medians[NUM_THREADS];

    for (int t = 0; t < NUM_THREADS; t++) {
        unsigned long long half = (args[t].end - args[t].start) / 2;
        unsigned long long cum  = 0;
        int mb = 0;
        for (int b = 0; b < NUM_BUCKETS; b++) {
            cum += args[t].bucket_counts[b];
            if (cum >= half) { mb = b; break; }
        }
        long long lo = RANGE_MIN +
            (long long)((unsigned long long)mb       * RANGE_SPAN / NUM_BUCKETS);
        long long hi = RANGE_MIN +
            (long long)((unsigned long long)(mb + 1) * RANGE_SPAN / NUM_BUCKETS);
        local_medians[t] = lo + (hi - lo) / 2;
        printf("  Thread %d -> %lld\n", t, local_medians[t]);
    }

    /* Sort local medians and pick middle */
    for (int i = 0; i < NUM_THREADS - 1; i++)
        for (int j = i + 1; j < NUM_THREADS; j++)
            if (local_medians[i] > local_medians[j]) {
                long long tmp      = local_medians[i];
                local_medians[i]   = local_medians[j];
                local_medians[j]   = tmp;
            }

    long long mom = local_medians[NUM_THREADS / 2];

    printf("\nMedian-of-Medians : %lld\n", mom);
    printf("Histogram Median  : %lld\n",   median_estimate);
    printf("\nNote: For Log-Normal(mu=0, sigma=0.8) shifted, the true median\n");
    printf("      of the raw distribution is e^0 = 1, mapped to RANGE_MIN + LN_SCALE ≈ %lld\n",
           (long long)(RANGE_MIN + LN_SCALE));

    return 0;
}
