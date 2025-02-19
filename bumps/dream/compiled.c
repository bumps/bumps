/*
Differential evolution MCMC stepper.
*/
#define _GNU_SOURCE  // sincos isn't standard?
#include <math.h>
#include <stdlib.h>

#include <stdio.h>  // for debugging

#ifdef _MSC_VER
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

// Random library with a separate generator for each thread of
// an OpenMP threaded program.  Assumes mafx 64 threads.  If OpenMP is
// not available, then operates single threaded.
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

// M_PI missing from MSVC math.h
#ifndef M_PI
#  define M_PI 3.141592653589793
#endif

// Limit to the number of threads so static thread-local data can be
// pre-allocated with the right size.
#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

// ==== Generator definition ====
// Uses:
//   Salmon, J. K.; Moraes, M. A.; Dror, R. O.; Shaw, D. E. (2011)
//   Parallel random numbers: as easy as 1, 2, 3.  In Proceedings of 2011
//   International Conference for High Performance Computing, Networking,
//   Storage and Analysis; SC '11; ACM: New York, NY; p 16:1016:12.
//   doi: 10.1145/2063384.2063405
//   https://www.deshawresearch.com/resources_random123.html  v1.09
// may want to swap it for a different generator, and update the following
#include <Random123/threefry.h>
typedef threefry4x64_ctr_t r123_ctr_t;
typedef threefry4x64_key_t r123_key_t;
typedef threefry4x64_ukey_t r123_ukey_t;
#define r123_init threefry4x64keyinit
#define r123_next threefry4x64
#define R123_SIZE 4          // the 4 in 4x64
typedef uint64_t randint_t;  // the 64 in 4x64
const randint_t R123_MAX = 18446744073709551615UL;
const double R123_TO_01 = 1.0/18446744073709551616.0;
const double R123_TO_M11 = 2.0/18446744073709551616.0;
// ==== end generator definition ====

typedef struct {
    r123_ctr_t counter; // position in sequence
    r123_key_t key;     // seed
    r123_ctr_t values;  // cached values not yet used
    int have_normal;    // Have a precomputer random normal
    double normal;      // the precomputed random normal
} Random;
Random streams[MAX_THREADS];  // Max of 64 different threads in OpenMP

double u_01_open(randint_t v) {
    return (((double)v) + 0.5)*R123_TO_01;
}

double u_m11_closed(randint_t v) {
    return ((double)((int64_t)v) + 0.5)*R123_TO_M11;
}

void _rand_init(randint_t seed)
{
    int thread_id = omp_get_thread_num();
    Random *rng = streams + thread_id;
    r123_ukey_t user_key;
    r123_key_t counter;
    int k;
    if (thread_id >= MAX_THREADS) {
        printf("Too many threads for random number generator.  Set OMP_NUM_THREADS=%d\n",
               MAX_THREADS);
        exit(1);
    }
    for (k = 0; k < R123_SIZE; k++) user_key.v[k] = counter.v[k] = 0;
    user_key.v[0] = seed;
    //user_key.v[1] = omp_get_thread_num();
    rng->key = r123_init(user_key);
    rng->counter = counter;
//printf("%d initializing %p with seed %llu and counter %llu\n", omp_get_thread_num(), rng, rng->key.v[0], rng->counter.v[0]);
    rng->have_normal = 0;
}

void rand_init(randint_t seed)
{
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    _rand_init(seed);
}

randint_t rand_next(void)
{
    Random *rng = streams+omp_get_thread_num();
//printf("retrieving from %p with key %ld and counter %ld\n",rng, rng->key.v[0], rng->counter.v[0]);
    if (rng->counter.v[0]%R123_SIZE == 0) {
        rng->values = r123_next(rng->counter, rng->key);
    }
    return rng->values.v[(rng->counter.v[0]++)%R123_SIZE];
}

double randn(void)
{
    Random *rng = &streams[omp_get_thread_num()];
    if (rng->have_normal) {
        rng->have_normal = 0;
        return rng->normal;
    } else {
        // Box-Muller transform converts two ints into two normals
        // Return one now and save the other for later.
        double x, y, r, arg;
        arg = M_PI*u_m11_closed(rand_next());
        x = sin(arg);
        y = cos(arg);
        r = sqrt(-2. * log(u_01_open(rand_next())));
        rng->have_normal = 1;
        rng->normal = y*r;
        return x*r;
    }
}

randint_t randint(randint_t range)
{
    while (1) {
        randint_t value = rand_next();
        // TODO: correct for very tiny bias against higher numbers.
        // Something like the following?
        //     if (value > R123_MAX-(R123_MAX%range)) continue;
        return value%range;
    }
}

double randu(void)
{
    return u_01_open(rand_next());
}

/* draw k unique from n objects not equal to q */
// Specialized for k << n.  If n is large and k -> n then argsort on
// a random uniform draw is a better bet.  If you don't want to exclude
// any numbers, set not_matching to total_num.
// TODO: raise an error instead of silently using replacement.
// The current behaviour is good enough for this code base, so not fixing here.
void rand_draw(int num_to_draw, int total_num, randint_t not_matching,
               randint_t p[])
{
    int i, j;
    // Handle the case where num_to_draw is too big
    if (num_to_draw > total_num - 1) {
        for (i = 0; i < total_num; i++) {
            p[i] = i;
        }
        num_to_draw -= total_num;
        p += total_num;
    }
    //printf("draw %d from %d != %llu\n", num_to_draw, total_num, not_matching);
    for (i=0; i < num_to_draw; i++) {
        while (1) {
            int proposed = randint(total_num);
            int unique = (proposed != not_matching);
            for (j=0; j < i && unique; j++) unique = (proposed != p[j]);
            if (unique) {
                p[i] = proposed;
                break;
            }
        }
    }
}

#if 0
#include <stdio.h>
#include <string.h>
#include <time.h>
randint_t random_seed()
{
    randint_t seed;
    FILE* urandom = fopen("/dev/urandom", "r");
    fread(&seed, sizeof(seed), 1, urandom);
    fclose(urandom);
    return seed;
}

void main(int argc, char *argv[])
{
    int j, k;
    randint_t seed, draw[10];
    seed = (argc == 1 ? random_seed() : atoi(argv[1]));
    printf("seed: %ld\n", seed);
    rand_init(seed);

    printf("i randint(10):\n");
    #pragma omp parallel for
    for (k=0; k < 10; k++) printf("i %d %ld\n", omp_get_thread_num(), randint(10));

    printf("u randu:\n");
    #pragma omp parallel for
    for (k=0; k < 10; k++) printf("u %d %g\n", omp_get_thread_num(), randu());

    printf("n randn:\n");
    #pragma omp parallel for
    for (k=0; k < 10; k++) printf("n %d %g\n", omp_get_thread_num(), randn());

    printf("d rand_draw(10,52,!5):\n");
    #pragma omp parallel for private(draw, j)
    for (k=0; k < 10; k++) {
        char buf[200];
        rand_draw(10, 52, 5, draw);
        sprintf(buf, "d %d ", omp_get_thread_num());
        for (j=0; j < 10; j++) sprintf(buf+strlen(buf), "%ld ", draw[j]);
        printf("%s\n", buf);
    }
}
#endif


#define _SNOOKER 0
#define _DE 1
#define _DIRECT 2

#define EPS 1e-6
#define MAX_CHAINS 20

/*
Generates offspring using METROPOLIS HASTINGS monte-carlo markov chain

The number of chains may be smaller than the population size if the
population is selected from both the current generation and the
ancestors.
*/
void
_perform_step(int qq, int Nchain, int Nvar, int NCR,
        double pop[], double CR[][2],
        int max_pairs, double eps,
        double snooker_rate, double de_rate, double noise, double scale,
        double x_new[], double step_alpha[], double CR_used[])
{
    randint_t chains[2*MAX_CHAINS];
    double u = randu();
    int alg = (u < snooker_rate ? _SNOOKER : u < de_rate ? _DE : _DIRECT);
    double *xin = &pop[qq*Nvar];
    int k;
//for (k=0; k < NCR; k++) printf("CR %d: %g %g\n", k, CR[k][0], CR[k][1]);
//printf("pop in c: ");
//for (k=0; k < Nvar; k++) printf("%g ", pop[qq*Nvar+k]);
//printf("\n");
//printf("alg: %d\n", alg);
    switch (alg) {
    case _DE: { // Use DE with cross-over ratio
        int var, num_crossover, active;
        double crossover_ratio, CR_cdf, distance, jiggle;

        // Select to number of vector pair differences to use in update
        // using k ~ discrete U[1, max pairs]
        int num_pairs = randint(max_pairs)+1;
        // [PAK: same as k = DEversion[qq, 1] in matlab version]

        // Weight the size of the jump inversely proportional to the
        // number of contributions, both from the parameters being
        // updated and from the population defining the step direction.
        double gamma_scale = 2.38/sqrt(2 * Nvar * num_pairs);
        // [PAK: same as F=Table_JumpRate[len(vars), k] in matlab version]

        // Select 2*k members at random different from the current member
        rand_draw(2*num_pairs, Nchain, qq, chains);

        // Select crossover ratio
        u = randu();
        CR_cdf = 0.;
        for (k=0; k < NCR-1; k++) {
            CR_cdf += CR[k][1];
            if (u <= CR_cdf) break;
        }
        crossover_ratio = CR[k][0];
        CR_used[qq] = crossover_ratio;

        // Select the dims to update based on the crossover ratio, making
        // sure at least one dim is selected
        num_crossover = 0;
        for (var=0; var < Nvar || num_crossover == 0; var++) {
            if (var == Nvar) {
                active = randint(Nvar);
            } else if (randu() <= crossover_ratio) {
                active = var;
            } else {
                x_new[var] = 0.;
                continue;
            }
            num_crossover++;

            // Find and average step from the selected pairs
            distance = 0.;
            for (k=0; k < num_pairs; k++) {
                distance += pop[chains[2*k]*Nvar + active] - pop[chains[2*k+1]*Nvar + active];
            }

            // Apply that step with F scaling and noise
            jiggle = 1 + eps * (2 * randu() - 1);
            x_new[active] = jiggle*gamma_scale*distance;
        }
        step_alpha[qq] = 1.;

        break;
    }

    case _SNOOKER: { // Use snooker update
        double num, denom, gamma_scale;

        // Select current and three others
        rand_draw(3, Nchain, qq, chains);
        double *z = &pop[chains[0]*Nvar];
        double *R1 = &pop[chains[1]*Nvar];
        double *R2 = &pop[chains[2]*Nvar];

        // Find the step direction and scale it to the length of the
        // projection of R1-R2 onto the step direction.
        // TODO: population sometimes not unique!
        for (k=0; k < Nvar; k++) x_new[k] = xin[k] - z[k];
        while (1) {
            denom = 0.; for (k=0; k < Nvar; k++) denom += x_new[k]*x_new[k];
            if (denom != 0.) break;
            for (k=0; k < Nvar; k++) x_new[k] = EPS*randn();
        }
        num = 0.; for (k=0; k < Nvar; k++) num += ((R1[k]-R2[k])*x_new[k]);

        // Step using gamma of 2.38/sqrt(2) + U(-0.5, 0.5)
        gamma_scale = (1.2 + randu())*num/denom;
        for (k=0; k < Nvar; k++) x_new[k] *= gamma_scale;

        // Scale Metropolis probability by (||xi* - z||/||xi - z||)^(d-1)
        num = 0.;
        for (k=0; k < Nvar; k++)
            num += (xin[k]+x_new[k]-z[k])*(xin[k]+x_new[k]-z[k]);
        step_alpha[qq] = pow(num/denom, (Nvar-1)/2);

        CR_used[qq] = 0.;
        break;
    }

    case _DIRECT: { // Use one pair and all dimensions
        // Note that there is no F scaling, dimension selection or noise
        int p[2];

        rand_draw(2, Nchain, qq, chains);
        double *R1 = &pop[chains[0]*Nvar];
        double *R2 = &pop[chains[1]*Nvar];
        for (k=0; k < Nvar; k++) x_new[k] = R1[k] - R2[k];
        step_alpha[qq] = 1.;
        CR_used[qq] = 0.;

        break;
    }
    }

//printf("alg %d -> ", alg);
//for (k=0; k < Nvar; k++) printf("%g ", x_new[k]);
//printf("\n");

    // Update x_old with delta_x and noise
    for (k=0; k < Nvar; k++) x_new[k] *= scale;

    // [PAK] The noise term needs to depend on the fitting range
    // of the parameter rather than using a fixed noise value for all
    // parameters.  The  current parameter value is a pretty good proxy
    // in most cases (i.e., relative noise), but it breaks down if the
    // parameter is zero, or if the range is something like 1 +/- eps.

    // absolute noise
    //for (k=0; k < Nvar; k++) x_new[k] += xin[k] + scale*noise*randn();

    // relative noise
    for (k=0; k < Nvar; k++) x_new[k] += xin[k]*(1.+scale*noise*randn());
//printf("alg %d -> ", alg);
//for (k=0; k < Nvar; k++) printf("%g ", x_new[k]);
//printf("\n");

    // no noise
    //for (k=0; k < Nvar; k++) x_new[k] += xin[k];

}

DLL_EXPORT void
de_step(int Nchain, int Nvar, int NCR,
        double pop[], double CR[][2],
        int max_pairs, double eps,
        double snooker_rate, double noise, double scale,
        double x_new[], double step_alpha[], double CR_used[])
{
    int qq;
    double de_rate = snooker_rate + 0.8 * (1-snooker_rate);

    //Choose snooker, de or direct according to snooker_rate, and 80:20
    // ratio of de to direct.

    //printf("in de_step with (%d,%d,%d) pairs=%d eps=%g snooker=%g noise=%g scale=%g\n",
    //Nchain, Nvar, NCR, max_pairs, eps, snooker_rate, noise, scale);
    //printf("points pop=%p CR=%p x_new=%p step_alpha=%p CR_used=%p\n", pop, CR, x_new, step_alpha, CR_used);

    // Chains evolve using information from other chains to create offspring
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (qq = 0; qq < Nchain; qq++) {
        _perform_step(qq, Nchain, Nvar, NCR, pop, CR,
                      max_pairs, eps, snooker_rate, de_rate,
                      noise, scale, &x_new[qq*Nvar], step_alpha, CR_used);
    }
}


DLL_EXPORT void
bounds_reflect(int Nchain, int Nvar, double pop[], double low[], double high[])
{
    int k, p, idx;

    #ifdef _OPENMP
    #pragma omp parallel for private(idx, k)
    #endif
    for (p=0; p < Nchain; p++) {
        for (k=0; k < Nvar; k++) {
            idx = p*Nvar+k;
            if (pop[idx] < low[k]) {
                pop[idx] = 2*low[k] - pop[idx];
            } else if (pop[idx] > high[k]) {
                pop[idx] = 2*high[k] - pop[idx];
            }
            if (pop[idx] < low[k] || pop[idx] > high[k]) {
                pop[idx] = low[k] + randu()*(high[k]-low[k]);
            }
        }
    }
}


DLL_EXPORT void
bounds_clip(int Nchain, int Nvar, double pop[], double low[], double high[])
{
    int k, p, idx;

    #ifdef _OPENMP
    #pragma omp parallel for private(idx, k)
    #endif
    for (p=0; p < Nchain; p++) {
        for (k=0; k < Nvar; k++) {
            idx = p*Nvar+k;
            if (pop[idx] < low[k]) {
                pop[idx] = low[k];
            } else if (pop[idx] > high[k]) {
                pop[idx] = high[k];
            }
        }
    }
}


DLL_EXPORT void
bounds_fold(int Nchain, int Nvar, double pop[], double low[], double high[])
{
    int k, p, idx;

    #ifdef _OPENMP
    #pragma omp parallel for private(idx, k)
    #endif
    for (p=0; p < Nchain; p++) {
        for (k=0; k < Nvar; k++) {
            idx = p*Nvar+k;
            if (pop[idx] < low[k]) {
                if (isinf(high[k])) {
                    pop[idx] = 2*low[k] - pop[idx];
                } else {
                    pop[idx] = high[k] - (low[k] - pop[idx]);
                }
            } else if (pop[idx] > high[k]) {
                if (isinf(low[k])) {
                    pop[idx] = 2*high[k] - pop[idx];
                } else {
                    pop[idx] = low[k] - (high[k] - pop[idx]);
                }
            }
            if (pop[idx] < low[k] || pop[idx] > high[k]) {
                pop[idx] = low[k] + randu()*(high[k]-low[k]);
            }
        }
    }
}


DLL_EXPORT void
bounds_random(int Nchain, int Nvar, double pop[], double low[], double high[])
{
    int k, p, idx;

    #ifdef _OPENMP
    #pragma omp parallel for private(idx, k)
    #endif
    for (p=0; p < Nchain; p++) {
        for (k=0; k < Nvar; k++) {
            idx = p*Nvar+k;
            if (pop[idx] < low[k]) {
                if (isinf(high[k])) {
                    pop[idx] = 2*low[k] - pop[idx];
                } else {
                    pop[idx] = low[k] + randu()*(high[k]-low[k]);
                }
            } else if (pop[idx] > high[k]) {
                if (isinf(low[k])) {
                    pop[idx] = 2*high[k] - pop[idx];
                } else {
                    pop[idx] = low[k] + randu()*(high[k]-low[k]);
                }
            }
        }
    }
}


DLL_EXPORT void
bounds_ignore(int Nchain, int Nvar, double pop[], double low[], double high[])
{
}
