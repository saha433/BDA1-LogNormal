/*
 * BDA Assignment-1 Q1: Min, Max, Mean — Log-Normal Distribution
 * Total elements : 2^40
 * Range          : [0, 10^9]
 * Compile: gcc -O2 -o q1_lognormal q1_lognormal.c -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

#define NUM_THREADS 8
#define TOTAL_ELEMENTS (1ULL << 40)  /* 2^40 */
#define MAX_VAL        1000000000ULL /* 10^9 */

/*
 * Log-Normal parameters (mu, sigma of the underlying normal).
 * We want the median of X = e^mu to land near 5e8 (mid-range).
 *   mu    = ln(5e8) ≈ 20.03
 *   sigma = 0.5  => moderate right-skew while keeping most mass in [0,10^9]
 */
#define LN_MU    20.03
#define LN_SIGMA  0.5

typedef struct {
    int      thread_id;
    uint64_t count;
    uint64_t local_min;
    uint64_t local_max;
    __uint128_t local_sum; /* 128-bit avoids overflow for 2^40 * 10^9 */
} ThreadArg;

/* Box-Muller: one standard normal sample */
static inline double std_normal(unsigned int *seed)
{
    double u1, u2;
    do { u1 = (double)rand_r(seed) / RAND_MAX; } while (u1 == 0.0);
    u2 = (double)rand_r(seed) / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Log-Normal sample clamped to [0, MAX_VAL] */
static inline uint64_t lognormal_sample(unsigned int *seed)
{
    double z   = std_normal(seed);
    double val = exp(LN_MU + LN_SIGMA * z);
    if (val < 0.0)         return 0;
    if (val > (double)MAX_VAL) return MAX_VAL;
    return (uint64_t)val;
}

void *thread_func(void *arg)
{
    ThreadArg *t = (ThreadArg *)arg;
    unsigned int seed = (unsigned int)(t->thread_id * 123456789UL + 987654321UL);

    uint64_t    lmin = UINT64_MAX;
    uint64_t    lmax = 0;
    __uint128_t lsum = 0;

    for (uint64_t i = 0; i < t->count; i++) {
        uint64_t v = lognormal_sample(&seed);
        if (v < lmin) lmin = v;
        if (v > lmax) lmax = v;
        lsum += v;
    }

    t->local_min = lmin;
    t->local_max = lmax;
    t->local_sum = lsum;
    return NULL;
}

int main(void)
{
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    uint64_t chunk = TOTAL_ELEMENTS / NUM_THREADS;

    printf("BDA Assignment-1 Q1: Min, Max, Mean (Log-Normal)\n");
    printf("Total elements : 2^40 = %llu\n", (unsigned long long)TOTAL_ELEMENTS);
    printf("Distribution   : Log-Normal(mu=%.2f, sigma=%.2f) clamped to [0, 10^9]\n",
           LN_MU, LN_SIGMA);
    printf("Theoretical    : Median ≈ e^mu = %.0f, Mean ≈ e^(mu+sigma^2/2) = %.0f\n",
           exp(LN_MU), exp(LN_MU + LN_SIGMA*LN_SIGMA/2.0));
    printf("Threads        : %d\n\n", NUM_THREADS);

    clock_t t0 = clock();

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].count     = (i == NUM_THREADS - 1)
                            ? (TOTAL_ELEMENTS - (uint64_t)i * chunk)
                            : chunk;
        pthread_create(&threads[i], NULL, thread_func, &args[i]);
    }

    uint64_t    gmin = UINT64_MAX, gmax = 0;
    __uint128_t gsum = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (args[i].local_min < gmin) gmin = args[i].local_min;
        if (args[i].local_max > gmax) gmax = args[i].local_max;
        gsum += args[i].local_sum;
    }

    double mean = (double)gsum / (double)TOTAL_ELEMENTS;

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("===== RESULTS =====\n");
    printf("Minimum : %llu\n",   (unsigned long long)gmin);
    printf("Maximum : %llu\n",   (unsigned long long)gmax);
    printf("Mean    : %.6f\n",   mean);
    printf("Time    : %.2f seconds\n", elapsed);

    return 0;
}
