/*
 * BDA Assignment-1 Q2: Dot Product & Cross Product — Log-Normal Distribution
 * Vector length : 10^10
 * Elements      : random integers from {-1, 0, 1}
 *
 * Log-Normal approach: generate X ~ LogNormal(mu=0, sigma=1).
 *   Map to {-1,0,1} by thresholding the normalised value:
 *     val < 0.5  -> -1  |  0.5 <= val <= 2.0  -> 0  |  val > 2.0  -> 1
 *   (median of LogNormal(0,1) = 1.0, so roughly symmetric mapping)
 *
 * Compile: gcc -O2 -o q2_lognormal q2_lognormal.c -lm -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <time.h>

#define NUM_THREADS 8
#define VECTOR_LEN  10000000000ULL /* 10^10 */

/* Underlying normal parameters for the log-normal */
#define LN_MU    0.0
#define LN_SIGMA 1.0

typedef struct {
    int     thread_id;
    uint64_t count;
    int64_t  local_dot;
    /* first 3 elements of each vector (captured by thread 0) */
    int a0, a1, a2;
    int b0, b1, b2;
} ThreadArg;

/* Box-Muller: one standard normal */
static inline double std_normal(unsigned int *seed)
{
    double u1, u2;
    do { u1 = (double)rand_r(seed) / RAND_MAX; } while (u1 == 0.0);
    u2 = (double)rand_r(seed) / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Log-Normal sample mapped to {-1, 0, 1} */
static inline int lognormal_discrete(unsigned int *seed)
{
    double x = exp(LN_MU + LN_SIGMA * std_normal(seed)); /* x > 0 */
    if (x < 0.5)  return -1;
    if (x > 2.0)  return  1;
    return 0;
}

void *thread_func(void *arg)
{
    ThreadArg *t = (ThreadArg *)arg;
    unsigned int seedA = (unsigned int)(t->thread_id * 111111111UL + 1);
    unsigned int seedB = (unsigned int)(t->thread_id * 999999999UL + 2);

    int64_t dot = 0;

    for (uint64_t i = 0; i < t->count; i++) {
        int a = lognormal_discrete(&seedA);
        int b = lognormal_discrete(&seedB);

        /* capture first 3 elements from thread 0 for cross product */
        if (t->thread_id == 0 && i < 3) {
            if (i == 0) { t->a0 = a; t->b0 = b; }
            if (i == 1) { t->a1 = a; t->b1 = b; }
            if (i == 2) { t->a2 = a; t->b2 = b; }
        }

        dot += (int64_t)a * b;
    }

    t->local_dot = dot;
    return NULL;
}

int main(void)
{
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    uint64_t chunk = VECTOR_LEN / NUM_THREADS;

    printf("BDA Assignment-1 Q2: Dot & Cross Products (Log-Normal)\n");
    printf("Vector length  : 10^10 = %llu\n", (unsigned long long)VECTOR_LEN);
    printf("Distribution   : LogNormal(mu=%.1f, sigma=%.1f) mapped to {-1,0,1}\n",
           LN_MU, LN_SIGMA);
    printf("Mapping        : x<0.5 -> -1 | 0.5<=x<=2.0 -> 0 | x>2.0 -> 1\n");
    printf("Threads        : %d\n\n", NUM_THREADS);
    printf("Processing...\n");

    clock_t t0 = clock();

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].count     = (i == NUM_THREADS - 1)
                            ? (VECTOR_LEN - (uint64_t)i * chunk)
                            : chunk;
        pthread_create(&threads[i], NULL, thread_func, &args[i]);
    }

    int64_t gdot = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        gdot += args[i].local_dot;
    }

    /* 3-D Cross product using first 3 elements from thread 0 */
    int a0 = args[0].a0, a1 = args[0].a1, a2 = args[0].a2;
    int b0 = args[0].b0, b1 = args[0].b1, b2 = args[0].b2;
    int cx = a1*b2 - a2*b1;
    int cy = a2*b0 - a0*b2;
    int cz = a0*b1 - a1*b0;

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

    printf("\n===== RESULTS =====\n");
    printf("Dot Product (sum over 10^10 elements) : %lld\n", (long long)gdot);
    printf("\n3-D Cross Product (using first 3 elements of each vector):\n");
    printf("  A = (%d, %d, %d)\n", a0, a1, a2);
    printf("  B = (%d, %d, %d)\n", b0, b1, b2);
    printf("  A x B = (%d, %d, %d)\n", cx, cy, cz);
    printf("\nTime : %.2f seconds\n", elapsed);

    return 0;
}
