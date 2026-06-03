/*
	wieferich.cl
	Bryan Little May 2026

	test a 96 bit number with a combined Fermat base 2 PRP test and Wieferich test

	Original 64 bit algorithm by
	Yves Gallot, Dec 2020
*/

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void wieferich(__global const cl_uint96_t *g_prime, __global result *g_result,
			__global uint *g_primecount, __global cl_uint96_t *g_cksum, __global ulong *g_total_pcnt){

	uint pcnt = g_primecount[0];
	uint lid = get_local_id(0);
	uint ls = get_local_size(0);
	uint group = get_group_id(0);
	uint gid = get_global_id(0);

	if(group * ls >= pcnt) return;	// no work for any thread in group

	__local cl_uint96_t sum[256];
	sum[lid] = zero96();

	__local uint fermat_count;
	if(!lid) fermat_count = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	if(gid < pcnt){
		cl_uint96_t p = g_prime[gid];
		cl_uint96_t q = mont_inv_pos96(p);

		const m2p96_t one = m2p_one96(p);
		const cl_uint96_t one_int = one96();
		const cl_uint96_t p_minus_1 = sub96_wrap(p, one_int);

		const m2p96_t fermat_mont = pow2_m2p96(p_minus_1, one, p, q);
		const uint fermat_pass = eq96(fermat_mont.r0, one.r0);

		if(fermat_pass){
			atomic_inc(&fermat_count);
			const cl_uint96_t quotient_mont = sub_mod96(fermat_mont.r1, one.r1, p);
			if (is_zero96(quotient_mont)) {
				uint i = atomic_inc(&g_primecount[2]);
				result ro = {p.lo, p.mid, p.hi, 0};
				g_result[i] = ro;
			}
			else{
				const cl_uint96_t threshold = make96_3(SPECIAL_THRESHOLD, 0U, 0U);
				const cl_uint96_t p_minus_threshold = sub96_wrap(p, threshold);
				const cl_uint96_t quotient = mont_get_modp96(quotient_mont, p, q);
				sum[lid] = quotient;

				if(le96(quotient, threshold)){
					int xr = small96_to_int(quotient);
					uint i = atomic_inc(&g_primecount[2]);
					result ro = {p.lo, p.mid, p.hi, xr};
					g_result[i] = ro;
				}
				else if(ge96(quotient, p_minus_threshold)){
					const cl_uint96_t delta = sub96_wrap(p, quotient);
					int xr = -small96_to_int(delta);
					uint i = atomic_inc(&g_primecount[2]);
					result ro = {p.lo, p.mid, p.hi, xr};
					g_result[i] = ro;
				}
			}
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			sum[lid] = add96_wrap( sum[lid], sum[lid+s] );
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(!lid){
		g_cksum[group] = add96_wrap(g_cksum[group], sum[0]);
		uint offset = group + 1;
		g_total_pcnt[offset] = g_total_pcnt[offset] + fermat_count; 
	}

	if(!gid){
		// add primecount to total primecount
		g_total_pcnt[0] += pcnt;
		// store largest kernel prime count for array bounds check
		if( pcnt > g_primecount[1] ){
			g_primecount[1] = pcnt;
		}
	}


}














