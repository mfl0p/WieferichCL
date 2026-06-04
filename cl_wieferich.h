// cl_wieferich.h

#define SPECIAL_THRESHOLD_MAX 1000000000U

typedef struct {
	int32_t fermat_pass;
	int32_t quot;
	int32_t euler_sign;
} wieferich_cpu_result_t;

typedef struct {
	uint32_t lo;
	uint32_t mid;
	uint32_t hi;
} cl_uint96_t;

typedef struct {
	uint32_t lo;
	uint32_t mid;
	uint32_t hi;
	int32_t euler;
	int32_t special;
} result;

typedef struct {
	__uint128_t pmin, pmax, p;
	uint64_t primecount, resultcount, last_trickle, state_sum, sieved;
	cl_uint96_t checksum;
	uint32_t threshold;
} workStatus;

typedef struct {
	uint32_t numgroups, computeunits, numresults, psize;
	uint64_t maxalloc, range;
	bool test, write_state_a_next, compute;
	const char *result_file;
} searchData;

typedef struct {
	uint32_t *h_primecount;
	uint64_t *h_total_pcnt;
	cl_uint96_t *h_cksum;
	cl_mem d_result;
	cl_mem d_cksum;
	cl_mem d_total_pcnt;
	cl_mem d_primecount;
	cl_mem d_prime;
	sclSoft clearresult, getsegprimes, clearn, wieferich;
} progData;

void cl_wieferich( sclHard hardware, workStatus & st, searchData & sd );

void run_test( sclHard hardware, workStatus & st, searchData & sd );

