/*
	WieferichCL
	by Bryan Little, May 2026
	and Yves Gallot

	Required minimum OpenCL version is 1.1
	CL_TARGET_OPENCL_VERSION 110 in simpleCL.h

	Search limits:  11 <= P < 2^96
	
*/

#ifdef _WIN32
#include "C:\msys64\usr\bin\include\pari\pari.h"
#undef long
#undef labs
#include "gmpwin.h"
#else
#include "/home/bryan/paribuild/include/pari/pari.h"
//#include "/home/bryan/pari-2.17.3/src/headers/pari.h"
#include "gmp.h"
#endif

#include <unistd.h>
#include <cinttypes>
#include <math.h>
#include <ctime>

#include "simpleCL.h"

#include "clearresult.h"
#include "getsegprimes.h"
#include "clearn.h"
#include "wieferich.h"
#include "common.h"

#include "cl_wieferich.h"

#include "boinc_api.h"
#include "boinc_opencl.h"

#define STATE_FILENAME_A "stateA.ckp"
#define STATE_FILENAME_B "stateB.ckp"

#define BITMAP_WORDS 4096
#define BITMAP_BITS (BITMAP_WORDS * 32)
#define BITMAP_RANGE (BITMAP_BITS * 2)
#define GPU_WORK_MULTIPLIER 12

void handle_trickle_up(workStatus & st){
	if(boinc_is_standalone()) return;
	uint64_t now = (uint64_t)time(NULL);
	if( (now-st.last_trickle) > 86400 ){	// Once per day
		st.last_trickle = now;
		double progress = boinc_get_fraction_done();
		double cpu;
		boinc_wu_cpu_time(cpu);
		APP_INIT_DATA init_data;
		boinc_get_init_data(init_data);
		double run = boinc_elapsed_time() + init_data.starting_elapsed_time;
		char msg[512];
		sprintf(msg, "<trickle_up>\n"
			"   <progress>%lf</progress>\n"
			"   <cputime>%lf</cputime>\n"
			"   <runtime>%lf</runtime>\n"
			"</trickle_up>\n",
			progress, cpu, run  );
		char variety[64];
		sprintf(variety, "ww_progress");
		boinc_send_trickle_up(variety, msg);
	}
}

void format_rate(double val, char *buf)
{
	const char *units[] = {"", "k", "M", "G", "T", "P"};
	int u = 0;

	while(val >= 1000.0 && u < 5){
		val /= 1000.0;
		u++;
	}

	sprintf(buf, "%6.2f %sp/s", val, units[u]);
}

void format_eta(double seconds, char *buf)
{
	int h = seconds / 3600;
	int m = ((int)seconds % 3600) / 60;
	int s = (int)seconds % 60;

	sprintf(buf, "%02d:%02d:%02d", h, m, s);
}

void print_progress(workStatus &st,
	double *smooth_rate,
	time_t start_time,
	__uint128_t run_start_p){
	const int bar_width = 40;

	__uint128_t total_range = st.pmax - st.pmin;
	__uint128_t done_total  = st.p - st.pmin;

	double progress = 0.0;
	if(total_range > 0)
		progress = (double)done_total / (double)total_range;

	if(progress < 0) progress = 0;
	if(progress > 1) progress = 1;

	int pos = (int)(progress * bar_width);

	time_t now = time(NULL);
	double elapsed = difftime(now, start_time);

	__uint128_t done_this_run = 0;
	if(st.p > run_start_p)
		done_this_run = st.p - run_start_p;

	double inst_rate = (double)done_this_run / (elapsed > 0 ? elapsed : 1.0);

	/*
	Exponential smoothing.
		*/
	if(*smooth_rate == 0.0)
		*smooth_rate = inst_rate;
	else
		*smooth_rate = 0.9 * (*smooth_rate) + 0.1 * inst_rate;

	__uint128_t remaining = 0;
	if(st.pmax > st.p)
		remaining = st.pmax - st.p;

	double remain = (double)remaining / (*smooth_rate > 0.0 ? *smooth_rate : 1.0);

	char rate_str[32];
	char eta_str[32];

	format_rate(*smooth_rate, rate_str);
	format_eta(remain, eta_str);

	printf("\r[");

	for(int i = 0; i < bar_width; i++)
		putchar(i < pos ? '#' : '-');

	printf("] %6.2f%% | %s | ETA %s",
		progress * 100.0,
		rate_str,
		eta_str);

	fflush(stdout);
}

FILE *my_fopen(const char * filename, const char * mode){
	char resolved_name[512];
	boinc_resolve_filename(filename,resolved_name,sizeof(resolved_name));
	return boinc_fopen(resolved_name,mode);
}


void cleanup( progData & pd, searchData & sd, workStatus & st ){
	sclReleaseMemObject(pd.d_result);
	sclReleaseMemObject(pd.d_cksum);
	sclReleaseMemObject(pd.d_primecount);
	sclReleaseMemObject(pd.d_total_pcnt);
	sclReleaseMemObject(pd.d_prime);

	sclReleaseClSoft(pd.clearresult);
	sclReleaseClSoft(pd.clearn);
	sclReleaseClSoft(pd.getsegprimes);
	sclReleaseClSoft(pd.wieferich);

	free(pd.h_primecount);
	free(pd.h_cksum);
	free(pd.h_total_pcnt);
}

// using fast binary checkpoint files with checksum calculation
void write_state( workStatus & st, searchData & sd ){

	FILE * out;

	st.state_sum = st.pmin+st.pmax+st.p+st.primecount+st.resultcount+st.last_trickle
		+st.sieved+st.checksum.lo+st.checksum.mid+st.checksum.hi+st.threshold;

	if (sd.write_state_a_next){
		if ((out = my_fopen(STATE_FILENAME_A,"wb")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_A);
	}
	else{
		if ((out = my_fopen(STATE_FILENAME_B,"wb")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_B);
	}

	if(out != NULL){

		if( fwrite(&st, sizeof(workStatus), 1, out) != 1 ){
			fprintf(stderr,"Cannot write checkpoint to file. Continuing...\n");
			// Attempt to close, even though we failed to write
			fclose(out);
		}
		else{
			// If state file is closed OK, write to the other state file
			// next time around
			if (fclose(out) == 0)
				sd.write_state_a_next = !sd.write_state_a_next;
		}
	}
}

int read_state( workStatus & st, searchData & sd ){

	FILE * in;
	bool good_state_a = true;
	bool good_state_b = true;
	workStatus stat_a, stat_b;

	// Attempt to read state file A
	if ((in = my_fopen(STATE_FILENAME_A,"rb")) == NULL){
		good_state_a = false;
	}
	else{
		if( fread(&stat_a, sizeof(workStatus), 1, in) != 1 ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_A);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else if(stat_a.pmin != st.pmin || stat_a.pmax != st.pmax){
			fprintf(stderr,"Invalid checkpoint file %s !!!\n",STATE_FILENAME_A);
			printf("Invalid checkpoint file %s !!!\n",STATE_FILENAME_A);
			good_state_a = false;
		}
		else{
			uint64_t state_sum = stat_a.pmin+stat_a.pmax+stat_a.p+stat_a.primecount+stat_a.resultcount+stat_a.last_trickle
				+stat_a.sieved+stat_a.checksum.lo+stat_a.checksum.mid+stat_a.checksum.hi+stat_a.threshold;
			if(state_sum != stat_a.state_sum){
				fprintf(stderr,"Checksum error in %s !!!\n",STATE_FILENAME_A);
				printf("Checksum error in %s !!!\n",STATE_FILENAME_A);
				good_state_a = false;
			}
		}
		fclose(in);
	}

	// Attempt to read state file B
	if ((in = my_fopen(STATE_FILENAME_B,"rb")) == NULL){
		good_state_b = false;
	}
	else{
		if( fread(&stat_b, sizeof(workStatus), 1, in) != 1 ){
			fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_B);
			printf("Cannot parse %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else if(stat_b.pmin != st.pmin || stat_b.pmax != st.pmax){
			fprintf(stderr,"Invalid checkpoint file %s !!!\n",STATE_FILENAME_B);
			printf("Invalid checkpoint file %s !!!\n",STATE_FILENAME_B);
			good_state_b = false;
		}
		else{
			uint64_t state_sum = stat_b.pmin+stat_b.pmax+stat_b.p+stat_b.primecount+stat_b.resultcount+stat_b.last_trickle
				+stat_b.sieved+stat_b.checksum.lo+stat_b.checksum.mid+stat_b.checksum.hi+stat_b.threshold;
			if(state_sum != stat_b.state_sum){
				fprintf(stderr,"Checksum error in %s !!!\n",STATE_FILENAME_B);
				printf("Checksum error in %s !!!\n",STATE_FILENAME_B);
				good_state_b = false;
			}
		}
		fclose(in);
	}

	// If both state files are OK, check which is the most recent
	if (good_state_a && good_state_b)
	{
		if (stat_a.p > stat_b.p)
			good_state_b = false;
		else
			good_state_a = false;
	}

	// Use data from the most recent state file
	if (good_state_a && !good_state_b)
	{
		memcpy(&st, &stat_a, sizeof(workStatus));
		sd.write_state_a_next = false;
		if(boinc_is_standalone()){
			printf("Resuming from checkpoint in %s\n",STATE_FILENAME_A);
		}
		return 1;
	}
	if (good_state_b && !good_state_a)
	{
		memcpy(&st, &stat_b, sizeof(workStatus));
		sd.write_state_a_next = true;
		if(boinc_is_standalone()){
			printf("Resuming from checkpoint in %s\n",STATE_FILENAME_B);
		}
		return 1;
	}

	// If we got here, neither state file was good
	return 0;
}

// sleep CPU thread while waiting on the specified event to complete in the command queue
void waitOnEvent(sclHard hardware, cl_event event)
{
	cl_int err;
	cl_int info;

#ifndef _WIN32
	struct timespec sleep_time;
	sleep_time.tv_sec  = 0;
	sleep_time.tv_nsec = 1000000; // 1ms
#endif

	boinc_begin_critical_section();

	err = clFlush(hardware.queue);
	if (err != CL_SUCCESS) {
		printf("ERROR: clFlush\n");
		fprintf(stderr, "ERROR: clFlush\n");
		sclPrintErrorFlags(err);
	}

	while (true) {

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time, NULL);
#endif

		err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
			sizeof(cl_int), &info, NULL);
		if (err != CL_SUCCESS) {
			printf("ERROR: clGetEventInfo\n");
			fprintf(stderr, "ERROR: clGetEventInfo\n");
			sclPrintErrorFlags(err);
		}

		if (info == CL_COMPLETE) {
			err = clReleaseEvent(event);
			if (err != CL_SUCCESS) {
				printf("ERROR: clReleaseEvent\n");
				fprintf(stderr, "ERROR: clReleaseEvent\n");
				sclPrintErrorFlags(err);
			}

			boinc_end_critical_section();
			return;
		}
	}
}


// queue a marker and sleep CPU thread until marker has been reached in the command queue
void sleepCPU(sclHard hardware){

	cl_event kernelsDone;
	cl_int err;
	cl_int info;
#ifndef _WIN32
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	// OpenCL v2.0
	/*
	err = clEnqueueMarkerWithWaitList( hardware.queue, 0, NULL, &kernelsDone);
	if ( err != CL_SUCCESS ) {
	printf( "ERROR: clEnqueueMarkerWithWaitList\n");
	fprintf(stderr, "ERROR: clEnqueueMarkerWithWaitList\n");
	sclPrintErrorFlags(err);
	}
		*/
	err = clEnqueueMarker( hardware.queue, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarker\n");
		fprintf(stderr, "ERROR: clEnqueueMarker\n");
		sclPrintErrorFlags(err);
	}

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(kernelsDone, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
		}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(kernelsDone);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
			}

			boinc_end_critical_section();

			return;
		}
	}
}

static void mpz_set_uint128(mpz_t z, __uint128_t x)
{
    uint64_t lo64 = (uint64_t)x;
    uint64_t hi64 = (uint64_t)(x >> 64);

    // Split lower 64 bits into two 32-bit halves
    unsigned long lo_lo = (unsigned long)(lo64 & 0xFFFFFFFF);
    unsigned long lo_hi = (unsigned long)(lo64 >> 32);

    // Split upper 64 bits into two 32-bit halves
    unsigned long hi_lo = (unsigned long)(hi64 & 0xFFFFFFFF);
    unsigned long hi_hi = (unsigned long)(hi64 >> 32);

    mpz_set_ui(z, hi_hi);           // topmost 32 bits
    mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, hi_lo);        // next 32 bits
    mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, lo_hi);        // next 32 bits
    mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, lo_lo);        // lowest 32 bits
}

void gmp_wieferich_fermat128(__uint128_t p, wieferich_cpu_result_t *out, uint32_t SPECIAL_THRESHOLD)
{
	mpz_t P;
	mpz_t P2;
	mpz_t E;
	mpz_t SP;
	mpz_t R0;
	mpz_t R1;
	mpz_t TWO;
	mpz_t threshold;
	mpz_t p_minus_threshold;
	mpz_t delta;

	mpz_inits(P, P2, E, SP, R0, R1, TWO,threshold, p_minus_threshold, delta, NULL);

	mpz_set_uint128(P, p);

	/* p^2 */
	mpz_mul(P2, P, P);

	/* e = p - 1 */
	mpz_sub_ui(E, P, 1);

	mpz_set_ui(TWO, 2);

	/*
	SP = 2^(p - 1) mod p^2
		*/
	mpz_powm(SP, TWO, E, P2);

	/*
	SP = R0 + p * R1

	R0 is the ordinary mod-p Fermat residue.
	R1 is the Fermat / Wieferich quotient modulo p.
		*/
	mpz_fdiv_qr(R1, R0, SP, P);

	memset(out, 0, sizeof(*out));

	out->fermat_pass = (mpz_cmp_ui(R0, 1) == 0);

	/*
	Match the OpenCL result-array behavior:

	if Fermat passes and sp.r1 == 0:
	quot = 0

	else if sp.r1 <= SPECIAL_THRESHOLD:
	quot = sp.r1

	else if sp.r1 >= p - SPECIAL_THRESHOLD:
	quot = -(p - sp.r1)
		*/
	if(out->fermat_pass)
	{
		mpz_set_ui(threshold, SPECIAL_THRESHOLD);
		mpz_sub(p_minus_threshold, P, threshold);

		if(mpz_cmp_ui(R1, 0) == 0)
		{
			out->quot = 0;
		}
		else if(mpz_cmp(R1, threshold) <= 0)
		{
			out->quot = (int)mpz_get_ui(R1);
		}
		else if(mpz_cmp(R1, p_minus_threshold) >= 0)
		{
			mpz_sub(delta, P, R1);
			out->quot = -(int)mpz_get_ui(delta);
		}
	}

	mpz_clears(P, P2, E, SP, R0, R1, TWO, threshold, p_minus_threshold, delta, NULL);

}

static const char *uint128_to_str(__uint128_t x, char buf[64])
{
	int i = 63;

	buf[i] = '\0';

	if (x == 0)
	{
		buf[--i] = '0';
		return &buf[i];
	}

	while (x != 0 && i > 0)
	{
		buf[--i] = (char)('0' + (unsigned)(x % 10));
		x /= 10;
	}

	return &buf[i];
}


static GEN uint128_to_pari(__uint128_t x)
{
	/*
	Build:
	(((w3 << 32) + w2) << 32 + w1) << 32 + w0

	This avoids relying on unsigned long being 64-bit.
		*/
	uint32_t w0 = (uint32_t)(x);
	uint32_t w1 = (uint32_t)(x >> 32);
	uint32_t w2 = (uint32_t)(x >> 64);
	uint32_t w3 = (uint32_t)(x >> 96);

	GEN n = utoi((ulong)w3);
	n = addii(shifti(n, 32), utoi((ulong)w2));
	n = addii(shifti(n, 32), utoi((ulong)w1));
	n = addii(shifti(n, 32), utoi((ulong)w0));

	return n;
}

static int pari_is_prime_u128(__uint128_t x)
{
	pari_sp av = avma;              // save PARI stack pointer

	GEN n = uint128_to_pari(x);
	GEN r = gisprime(n, 0);         // exact primality test

	int is_prime = !gequal0(r);

	avma = av;                      // clear temporary PARI objects
	return is_prime;
}

static inline __uint128_t u96_to_u128(const cl_uint96_t x)
{
	return ((__uint128_t)x.hi  << 64) | ((__uint128_t)x.mid << 32) | (__uint128_t)x.lo;
}

static inline cl_uint96_t u128_to_u96_wrap(const __uint128_t x)
{
	cl_uint96_t r;

	r.lo  = (uint32_t)x;
	r.mid = (uint32_t)(x >> 32);
	r.hi  = (uint32_t)(x >> 64);

	return r;
}

static inline cl_uint96_t add96_carry(const cl_uint96_t a, const cl_uint96_t b, cl_uint *carry)
{
	const __uint128_t s = u96_to_u128(a) + u96_to_u128(b);

	/*
		a and b are 96-bit, so s < 2^97.
		The carry out of bit 95 is therefore only 0 or 1.
	*/
	*carry = (uint32_t)(s >> 96);

	return u128_to_u96_wrap(s);
}

static inline cl_uint96_t add96_wrap(const cl_uint96_t a, const cl_uint96_t b)
{
	const __uint128_t s = u96_to_u128(a) + u96_to_u128(b);

	return u128_to_u96_wrap(s);
}

int resultcompare(const void *a, const void *b)
{
	const result *resA = (const result *)a;
	const result *resB = (const result *)b;

	const cl_uint96_t xa = {
		.lo  = resA->lo,
		.mid = resA->mid,
		.hi  = resA->hi
	};

	const cl_uint96_t xb = {
		.lo  = resB->lo,
		.mid = resB->mid,
		.hi  = resB->hi
	};

	const __uint128_t pa = u96_to_u128(xa);
	const __uint128_t pb = u96_to_u128(xb);

	if(pa > pb)
		return 1;
	if(pa < pb)
		return -1;

	return 0;
}

void checkpoint( workStatus & st, searchData & sd ){
	handle_trickle_up( st );
	write_state( st, sd );
	boinc_checkpoint_completed();
}

void getResults( progData & pd, workStatus & st, searchData & sd, sclHard hardware ){

	if(boinc_is_standalone()){
		printf("\r                                                                                \r");
		fflush(stdout);
	}

	// copy total prime count to host memory, non-blocking
	sclReadNB(hardware, sd.numgroups*sizeof(uint64_t), pd.d_total_pcnt, pd.h_total_pcnt);

	// copy checksum to host memory, non-blocking
	sclReadNB(hardware, sd.numgroups*sizeof(cl_uint96_t), pd.d_cksum, pd.h_cksum);

	// copy prime count to host memory, blocking
	sclRead(hardware, 3*sizeof(cl_uint), pd.d_primecount, pd.h_primecount);

	// sum group totals
	for(uint32_t i=0; i<sd.numgroups; ++i){
		st.checksum = add96_wrap( st.checksum, pd.h_cksum[i] );
		if(i>0)	st.primecount += pd.h_total_pcnt[i];
	}

	// index 0 is the number of candidates after sieve
	st.sieved += pd.h_total_pcnt[0];

	// largest kernel prime count.  used to check array bounds
	if(pd.h_primecount[1] > sd.psize){
		fprintf(stderr,"error: gpu prime array overflow\n");
		printf("error: gpu prime array overflow\n");
		exit(EXIT_FAILURE);
	}

	//	printf("pd.h_primecount[1]: %u\n",pd.h_primecount[1]);

	uint32_t resultcount = pd.h_primecount[2];

	if(resultcount){

		// used to check array bounds.
		// probably only works with a small overflow.
		// large overflow will crash the program.
		if(resultcount > sd.numresults){
			fprintf(stderr,"Error: number of results (%u) overflowed array.\n", resultcount);
			exit(EXIT_FAILURE);
		}

		result *h_result = (result *)malloc(resultcount * sizeof(result));
		if( h_result == NULL ){
			fprintf(stderr,"malloc error: h_result\n");
			exit(EXIT_FAILURE);
		}

		// copy results to host memory, blocking
		sclRead(hardware, resultcount * sizeof(result), pd.d_result, h_result);

		// sort results by prime size if needed
		if(resultcount > 1){
			qsort(h_result, resultcount, sizeof(result), resultcompare);
		}

		if(boinc_is_standalone()) printf("Verifying %u results on CPU...\n", resultcount);

		FILE *resfile = NULL;
		for(uint32_t i=0; i<resultcount; ++i){

			__uint128_t p = ((__uint128_t)h_result[i].hi  << 64) | ((__uint128_t)h_result[i].mid << 32) | (__uint128_t)h_result[i].lo;
			int special = h_result[i].special;
			
			wieferich_cpu_result_t r;
			gmp_wieferich_fermat128(p, &r, st.threshold);

			char buf[64];
			if(!r.fermat_pass || r.quot != special){
				printf("Error: GPU calculated invalid result: p %s quot %d\n", uint128_to_str(p,buf),special);
				printf("CPU calculated: fermat_pass %d quot %d\n", r.fermat_pass, r.quot);
				fprintf(stderr,"Error: GPU calculated invalid result: p %s quot %d\n", uint128_to_str(p,buf),special);
				fprintf(stderr,"CPU calculated: fermat_pass %d quot %d\n", r.fermat_pass, r.quot);
				exit(EXIT_FAILURE);
			}

			if( !pari_is_prime_u128(p) ){
				printf("PariGP: p %s is composite!\n", uint128_to_str(p,buf) );
				fprintf(stderr,"PariGP: p %s is composite!\n", uint128_to_str(p,buf) );
				continue;
			}

			st.resultcount++;

			// write results to file
			if(resfile == NULL){
				resfile = my_fopen(sd.result_file,"a");
				if( resfile == NULL ){
					fprintf(stderr,"Cannot open %s !!!\n",sd.result_file);
					exit(EXIT_FAILURE);
				}
			}

			if( fprintf( resfile, "%s %d\n",uint128_to_str(p,buf),special ) < 0 ){
				fprintf(stderr,"Cannot write to %s !!!\n",sd.result_file);
				exit(EXIT_FAILURE);
			}
		}

		if(resfile != NULL){
			fclose(resfile);
		}
		free(h_result);

		fprintf(stderr,"Verified %u results.\n", resultcount);
		if(boinc_is_standalone()) printf("Verified %u results.\n", resultcount);
	}

	checkpoint(st, sd);
}


void setupSearch(workStatus & st, searchData & sd){

	char buf[64];

	if(!st.pmin || !st.pmax){
		printf("\n-p and -P arguments are required\nuse -h for help\n");
		fprintf(stderr, "-p and -P arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if(st.pmin >= st.pmax){
		printf("pmin < pmax is required\nuse -h for help\n");
		fprintf(stderr, "pmin < pmax is required\n");
		exit(EXIT_FAILURE);
	}

	if(st.pmin <= UINT64_MAX){
		st.threshold = 10000;
	}

	st.p = st.pmin;

	if(boinc_is_standalone()){
		printf("Start p: %s\n", uint128_to_str(st.pmin,buf) );
		printf("  End P: %s\n", uint128_to_str(st.pmax,buf) );
		printf("Using near-Wieferich threshold of %u\n", st.threshold );
	}

	fprintf(stderr, "Start p: %s\n", uint128_to_str(st.pmin,buf) );
	fprintf(stderr, "  End P: %s\n", uint128_to_str(st.pmax,buf) );
	fprintf(stderr, "Using near-Wieferich threshold of %u\n", st.threshold );
}


void finalizeResults( workStatus & st, searchData & sd ){

	char line[256];
	uint32_t lc = 0;
	FILE * resfile;

	if(st.resultcount){
		// check result file has the same number of lines as the result count
		resfile = my_fopen(sd.result_file,"r");

		if(resfile == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",sd.result_file);
			exit(EXIT_FAILURE);
		}

		while(fgets(line, sizeof(line), resfile) != NULL) {
				++lc;
		}

		fclose(resfile);

		if(lc < st.resultcount){
			fprintf(stderr,"ERROR: Missing results in %s !!!\n",sd.result_file);
			printf("ERROR: Missing results in %s !!!\n",sd.result_file);
			exit(EXIT_FAILURE);
		}
	}

	// print checksum
	resfile = my_fopen(sd.result_file,"a");

	if(resfile == NULL){
		fprintf(stderr,"Cannot open %s !!!\n",sd.result_file);
		exit(EXIT_FAILURE);
	}

	if(st.resultcount){
		if( fprintf( resfile, "%08x%08x%08x\n", st.checksum.hi, st.checksum.mid, st.checksum.lo) < 0){
			fprintf(stderr,"Cannot write to %s !!!\n",sd.result_file);
			exit(EXIT_FAILURE);
		}
	}
	else{
		if( fprintf( resfile, "no results\n%08x%08x%08x\n", st.checksum.hi, st.checksum.mid, st.checksum.lo) < 0){
			fprintf(stderr,"Cannot write to %s !!!\n",sd.result_file);
			exit(EXIT_FAILURE);
		}
	}

	fclose(resfile);
}

// set values that change based on gpu work size
void resize(progData & pd, workStatus & st, searchData & sd, sclHard hardware){

	cl_int err = 0;

	// after sieving to 2^16, and range starting above 2^32, about 5.06133% candidates remain in range
	// at low p it may be closer to 8.0%  
	sd.psize = (uint32_t)((double)sd.range * 0.06);
	if(st.pmin < UINT_MAX) sd.psize *= 2;

	// number of gpu workgroups, used to size the checksum array on gpu
	sd.numgroups = (sd.psize / pd.wieferich.local_size[0]) + 2;

	// setup global sizes
	sclSetGlobalSize( pd.clearresult, sd.numgroups );
	sclSetGlobalSize( pd.wieferich, sd.psize );
	sclSetGlobalSize( pd.getsegprimes, sd.range / BITMAP_RANGE * pd.getsegprimes.local_size[0] );

	sclReleaseMemObject(pd.d_cksum);
	pd.d_cksum = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numgroups*sizeof(cl_uint96_t), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: d_sum array.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	sclReleaseMemObject(pd.d_total_pcnt);
	pd.d_total_pcnt = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numgroups*sizeof(cl_ulong), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: d_resultcount array.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	sclReleaseMemObject(pd.d_prime);
	pd.d_prime = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.psize*sizeof(cl_uint96_t), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: d_result array.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	// host arrays used for data transfer from gpu during checkpoints
	if(pd.h_cksum != NULL) free(pd.h_cksum);
	pd.h_cksum = (cl_uint96_t *)malloc(sd.numgroups*sizeof(cl_uint96_t));
	if( pd.h_cksum == NULL ){
		fprintf(stderr,"malloc error: h_cksum\n");
		exit(EXIT_FAILURE);
	}
	if(pd.h_total_pcnt != NULL) free(pd.h_total_pcnt);
	pd.h_total_pcnt = (uint64_t *)malloc(sd.numgroups*sizeof(uint64_t));
	if( pd.h_total_pcnt == NULL ){
		fprintf(stderr,"malloc error: h_total_pcnt\n");
		exit(EXIT_FAILURE);
	}

	// set static kernel args
	sclSetKernelArg(pd.clearresult, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.clearresult, 1, sizeof(cl_mem), &pd.d_total_pcnt);
	sclSetKernelArg(pd.clearresult, 2, sizeof(cl_mem), &pd.d_cksum);
	sclSetKernelArg(pd.clearresult, 3, sizeof(uint32_t), &sd.numgroups);

	sclSetKernelArg(pd.getsegprimes, 2, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.getsegprimes, 3, sizeof(cl_mem), &pd.d_prime);

	sclSetKernelArg(pd.wieferich, 0, sizeof(cl_mem), &pd.d_prime);
	sclSetKernelArg(pd.wieferich, 1, sizeof(cl_mem), &pd.d_result);
	sclSetKernelArg(pd.wieferich, 2, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.wieferich, 3, sizeof(cl_mem), &pd.d_cksum);
	sclSetKernelArg(pd.wieferich, 4, sizeof(cl_mem), &pd.d_total_pcnt);

}

void cl_wieferich( sclHard hardware, workStatus & st, searchData & sd ){

	progData pd = {};
	time_t boinc_last, ckpt_last, time_curr;
	cl_int err = 0;

	pari_init(8000000, 0);

	if(sd.result_file == NULL){
		sd.result_file = "results-WW.txt";
	}

	// setup kernel parameters
	setupSearch(st,sd);

	// build kernels
	pd.clearn = sclGetCLSoftware(clearn_cl,"clearn",hardware, NULL);
	pd.clearresult = sclGetCLSoftwareWithCommon(common_cl, clearresult_cl,"clearresult",hardware, NULL);
	// bake constants into kernel source
	char cldef[256];
	snprintf(cldef, sizeof(cldef), "-DBITMAP_WORDS=%d -DSPECIAL_THRESHOLD=%uU", BITMAP_WORDS, st.threshold);
	pd.wieferich = sclGetCLSoftwareWithCommon(common_cl, wieferich_cl,"wieferich",hardware, cldef);
	pd.getsegprimes = sclGetCLSoftwareWithCommon(common_cl, getsegprimes_cl,"getsegprimes",hardware, cldef);

	// kernels have __attribute__ ((reqd_work_group_size(256, 1, 1)))
	// it's still possible the CL complier picked a different size
	if(pd.wieferich.local_size[0] != 256){
		pd.wieferich.local_size[0] = 256;
		fprintf(stderr, "Set wieferich kernel local size to 256\n");
	}

	// resume from checkpoint or start new sieve
	if( sd.test ){
		// clear result file
		FILE * temp_file = my_fopen(sd.result_file,"w");
		if (temp_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",sd.result_file);
			exit(EXIT_FAILURE);
		}
		fclose(temp_file);
	}
	else{
		// Resume from checkpoint if there is one
		if( read_state( st, sd ) ){
			if(boinc_is_standalone()){
				char buf[64];
				printf("Resuming from checkpoint. Current p: %s\n", uint128_to_str(st.p, buf) );
			}
			fprintf(stderr,"Resuming from checkpoint\n");

			//trying to resume a finished workunit
			if( st.p == st.pmax ){
				if(boinc_is_standalone()){
					printf("Workunit complete.\n");
				}
				fprintf(stderr,"Workunit complete.\n");
				boinc_finish(EXIT_SUCCESS);
			}

		}
		// starting from beginning
		else{
			// clear result file
			FILE * temp_file = my_fopen(sd.result_file,"w");
			if (temp_file == NULL){
				fprintf(stderr,"Cannot open %s !!!\n",sd.result_file);
				exit(EXIT_FAILURE);
			}
			fclose(temp_file);

			// setup boinc trickle up... trickle 1 day after starting.
			st.last_trickle = (uint64_t)time(NULL);
		}
	}

	// Allocate mem
	pd.d_result = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numresults*sizeof(result), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: d_result array.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_primecount = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, 3*sizeof(cl_uint), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: d_primecount array.\n");
		printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.h_primecount = (uint32_t *)malloc(3*sizeof(uint32_t));
	if( pd.h_primecount == NULL ){
		fprintf(stderr,"malloc error: h_primecount\n");
		exit(EXIT_FAILURE);
	}

	// setup clearn kernel
	sclSetGlobalSize( pd.clearn, 64 );
	sclSetKernelArg(pd.clearn, 0, sizeof(cl_mem), &pd.d_primecount);

	// setup gpu work size
	sd.range = (uint64_t)sd.computeunits * GPU_WORK_MULTIPLIER * BITMAP_RANGE;
	if(sd.range > INT_MAX) sd.range = (INT_MAX / BITMAP_RANGE) * BITMAP_RANGE;
	resize(pd, st, sd, hardware);

	time(&boinc_last);
	time(&ckpt_last);

	time_t totals, totalf;
	if(boinc_is_standalone()){
		time(&totals);
	}

	time_t start_time = time(NULL);
	double smooth_rate = 0;
	__uint128_t run_start_p = st.p;
	bool first_iter = true;
	const double irsize = 1.0 / (double)(st.pmax-st.pmin);
	const int maxq = sd.compute ? 15 : 75;
	int kernelq = 0;
	cl_event launchEvent = NULL;

	sclEnqueueKernel(hardware, pd.clearresult);

	fprintf(stderr,"Starting Search...\n");
	if(boinc_is_standalone()){
		printf("Starting Search...\n");
	}

	// main search loop
	while(st.p < st.pmax){

		// clear gpu candidate counter
		if(!kernelq){
			launchEvent = sclEnqueueKernelEvent(hardware, pd.clearn);
		}
		else{
			sclEnqueueKernel(hardware, pd.clearn);
		}
		if(++kernelq == maxq){
			// limit cl queue depth and sleep cpu
			waitOnEvent(hardware, launchEvent);
			kernelq = 0;
		}

		// starting p must be odd
		if((st.p & 1) == 0) st.p++;

		__uint128_t stop = st.p + sd.range;
		if(stop > st.pmax) stop = st.pmax;

		cl_uint96_t kernel_start = u128_to_u96_wrap(st.p);
		cl_uint96_t kernel_stop = u128_to_u96_wrap(stop);
		st.p = stop;

		// get primes
		sclSetKernelArg(pd.getsegprimes, 0, sizeof(cl_uint96_t), &kernel_start);
		sclSetKernelArg(pd.getsegprimes, 1, sizeof(cl_uint96_t), &kernel_stop);

		//		float kernel_ms = ProfilesclEnqueueKernel(hardware, pd.getsegprimes);
		//		printf("getseg kernel time %0.2fms\n",kernel_ms);
		sclEnqueueKernel(hardware, pd.getsegprimes);

		// adjust gpu work size to prevent screen lag
		if(first_iter){
			first_iter = false;
			float kernel_ms = ProfilesclEnqueueKernel(hardware, pd.wieferich);
			getResults(pd, st, sd, hardware);
			sclEnqueueKernel(hardware, pd.clearresult);
			double multi = sd.compute ? 50.0/kernel_ms : 10.0/kernel_ms;	// target kernel time 50ms or 10ms
			sd.range = (uint64_t)( multi * (double)sd.range );
			if(sd.range > INT_MAX) sd.range = INT_MAX;
			if(sd.range < BITMAP_RANGE) sd.range = BITMAP_RANGE;
			sd.range = (sd.range / BITMAP_RANGE) * BITMAP_RANGE;
			resize(pd, st, sd, hardware);
			fprintf(stderr,"c:%u u:%u r:%u p:%u\n", (uint32_t)sd.compute, sd.computeunits, (uint32_t)sd.range, sd.psize);
			if(boinc_is_standalone()){
				printf("c:%u u:%u r:%u p:%u\n", (uint32_t)sd.compute, sd.computeunits, (uint32_t)sd.range, sd.psize);
			}
		}
		else{
			//			float wkernel_ms = ProfilesclEnqueueKernel(hardware, pd.wieferich);
			//			printf("wieferich kernel time %0.2fms\n",wkernel_ms);
			sclEnqueueKernel(hardware, pd.wieferich);
		}

		time(&time_curr);
		// update BOINC fraction done every 2 sec
		if( ((int)time_curr - (int)boinc_last) > 1 ){
			double fd = (double)(st.p-st.pmin)*irsize;
			boinc_fraction_done(fd);
			if(boinc_is_standalone()){
				print_progress(st, &smooth_rate, start_time, run_start_p);
			}
			boinc_last = time_curr;
		}

		// checkpoint at 1 minute
		if( ((int)time_curr - (int)ckpt_last) > 60 ){
			if(kernelq){
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
			sleepCPU(hardware);
			boinc_begin_critical_section();
			getResults(pd, st, sd, hardware);
			boinc_end_critical_section();
			ckpt_last = time_curr;
			sclEnqueueKernel(hardware, pd.clearresult);
			if(boinc_is_standalone()){
				char buf[64];
				printf("Checkpoint, current p: %s\n", uint128_to_str(st.p, buf) );
			}
		}

	}

	sleepCPU(hardware);

	boinc_begin_critical_section();
	st.p = st.pmax;
	boinc_fraction_done(1.0);
	getResults(pd, st, sd, hardware);
	finalizeResults(st,sd);
	boinc_end_critical_section();

	fprintf(stderr,"Search complete.\nresults %" PRIu64 ", prime count %" PRIu64 "\n", st.resultcount, st.primecount);

	if(boinc_is_standalone()){
		time(&totalf);
		printf("Search finished in %d sec.\n", (int)totalf - (int)totals);
		printf("results %" PRIu64 ", sieved %" PRIu64 ", prime count %" PRIu64 ", checksum %08x%08x%08x\n",
			st.resultcount, st.sieved, st.primecount, st.checksum.hi, st.checksum.mid, st.checksum.lo); 
	}

	// cleanup
	cleanup(pd, sd, st);

	pari_close();

}



void run_test( sclHard hardware, workStatus & st, searchData & sd ){

	fprintf(stderr,"self test not implemented yet!\n");
	printf("self test not implemented yet!\n");
	exit(EXIT_FAILURE);


}


