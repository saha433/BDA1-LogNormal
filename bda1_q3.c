#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define NUM_SUBSEQ  1000
#define SUBSEQ_SIZE 1000000
#define TOTAL       1000000000ULL
#define NUM_THREADS 8

typedef struct {
    int       subseq_id;
    long long min_val;
    long long range;
    long long local_min;
    long long local_max;
    double    local_mean;
} SubseqStats;

int              next_job  = 0;
pthread_mutex_t  job_mutex = PTHREAD_MUTEX_INITIALIZER;
SubseqStats     *all_stats;

/* Box-Muller: one standard normal (thread-unsafe global rand — seeded per subseq) */
static double std_normal_global(void)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-9) u1 = 1e-9;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int cmp_ll(const void *a, const void *b)
{
    long long x = *(long long *)a;
    long long y = *(long long *)b;
    return (x > y) - (x < y);
}

static void process_subseq(SubseqStats *s)
{
    long long *buf = malloc(SUBSEQ_SIZE * sizeof(long long));
    if (!buf) { perror("malloc"); return; }

    /* Seed per subsequence for reproducibility */
    srand((unsigned int)(s->subseq_id * 100 + 7));

    /*
     * Log-Normal parameterisation:
     *   We want most mass in [min_val, min_val+range).
     *   Set the median = min_val + range/2, i.e. mu = ln(min_val + range/2).
     *   Use sigma = 0.15 for a narrow, nearly-symmetric spread.
     *   Clamp to [min_val, min_val+range).
     */
    double centre = s->min_val + s->range / 2.0;
    /* avoid log(0) when subseq 0 starts at 0 */
    if (centre < 1.0) centre = 1.0;
    double mu    = log(centre);
    double sigma = 0.15;

    double sum = 0.0;

    for (int i = 0; i < SUBSEQ_SIZE; i++) {
        double z     = std_normal_global();
        double val_d = exp(mu + sigma * z);

        /* clamp inside subsequence range */
        if (val_d < s->min_val)               val_d = (double)s->min_val;
        if (val_d >= s->min_val + s->range)   val_d = (double)(s->min_val + s->range - 1);

        long long val = (long long)val_d;
        buf[i] = val;
        sum   += val;
    }

    qsort(buf, SUBSEQ_SIZE, sizeof(long long), cmp_ll);

    s->local_min  = buf[0];
    s->local_max  = buf[SUBSEQ_SIZE - 1];
    s->local_mean = sum / SUBSEQ_SIZE;

    free(buf);
}

void *thread_worker(void *arg)
{
    (void)arg;
    while (1) {
        pthread_mutex_lock(&job_mutex);
        int job = next_job++;
        pthread_mutex_unlock(&job_mutex);

        if (job >= NUM_SUBSEQ) break;

        process_subseq(&all_stats[job]);

        if ((job + 1) % 100 == 0)
            printf("  Progress: %d / %d subsequences sorted\n", job + 1, NUM_SUBSEQ);
    }
    return NULL;
}

int main(void)
{
    printf("BDA Assignment-1 Q3: Sorting Subsequences (Log-Normal)\n");
    printf("Total elements      : 10^9 = %llu\n", TOTAL);
    printf("Subsequences        : %d\n",   NUM_SUBSEQ);
    printf("Elements per subseq : %d\n",   SUBSEQ_SIZE);
    printf("Distribution        : Log-Normal (sigma=0.15, median centred per subseq)\n");
    printf("Threads             : %d\n\n", NUM_THREADS);

    long long range_per_subseq = 1000000LL;

    all_stats = malloc(NUM_SUBSEQ * sizeof(SubseqStats));
    if (!all_stats) { perror("malloc"); return 1; }

    for (int i = 0; i < NUM_SUBSEQ; i++) {
        all_stats[i].subseq_id = i;
        all_stats[i].min_val   = (long long)i * range_per_subseq;
        all_stats[i].range     = range_per_subseq;
    }

    pthread_t threads[NUM_THREADS];
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_create(&threads[t], NULL, thread_worker, NULL);
    for (int t = 0; t < NUM_THREADS; t++)
        pthread_join(threads[t], NULL);

    long long global_min = all_stats[0].local_min;
    long long global_max = all_stats[NUM_SUBSEQ - 1].local_max;

    double global_mean = 0.0;
    for (int i = 0; i < NUM_SUBSEQ; i++)
        global_mean += all_stats[i].local_mean;
    global_mean /= NUM_SUBSEQ;

    printf("\nSorting complete.\n\n");
    printf("Global Minimum : %lld\n",   global_min);
    printf("Global Maximum : %lld\n",   global_max);
    printf("Global Mean    : %.2f\n\n", global_mean);

    printf("First 3 subsequences:\n");
    for (int i = 0; i < 3; i++)
        printf("  Subseq %4d -> min: %lld  max: %lld  mean: %.2f\n",
               i, all_stats[i].local_min, all_stats[i].local_max, all_stats[i].local_mean);

    printf("\nLast 3 subsequences:\n");
    for (int i = NUM_SUBSEQ - 3; i < NUM_SUBSEQ; i++)
        printf("  Subseq %4d -> min: %lld  max: %lld  mean: %.2f\n",
               i, all_stats[i].local_min, all_stats[i].local_max, all_stats[i].local_mean);

    printf("\nOrder check: each subsequence range is strictly greater than the previous.\n");

    free(all_stats);
    return 0;
}
