/*
	wieferich.cl
	Bryan Little May 2026

	test a 96 bit number with a combined Fermat base 2 PRP test and Wieferich test

	Original 64 bit algorithm by
	Yves Gallot, Dec 2020
*/

static inline void store_result(cl_uint96_t p, int e, int s, __global uint *g_primecount, __global result *g_result){
	result res;
	res.lo = p.lo;
	res.mid = p.mid;
	res.hi = p.hi;
	res.euler = e;
	res.special = s;
	g_result[atomic_inc(&g_primecount[2])] = res;
}

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void wieferich(__global const cl_uint96_t *g_prime, __global result *g_result,
			__global uint *g_primecount, __global cl_uint96_t *g_cksum, __global ulong *g_total_pcnt){

	const uint pcnt = g_primecount[0];
	const uint lid = get_local_id(0);
	const uint ls = get_local_size(0);
	const uint group = get_group_id(0);
	const uint gid = get_global_id(0);

	if(group * ls >= pcnt) return;	// no work for any thread in group

	__local cl_uint96_t sum[256];
	sum[lid] = zero96();

	__local uint fermat_count;
	if(!lid) fermat_count = 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	if(gid < pcnt){
		const cl_uint96_t p = g_prime[gid];
		const cl_uint96_t q = mont_inv_pos96(p);
		const m2p96_t one = m2p_one96(p);
		const cl_uint96_t one_int = one96();
		const cl_uint96_t p_minus_1 = sub96_wrap(p, one_int);

		// fermat_mont is montgomerized 2^(p - 1) mod p^2  
		const m2p96_t fermat_mont = pow2_m2p96(p_minus_1, one, p, q);

		// if Fermat PRP test passes, 2^(p - 1) = 1 mod p^2, remainder = r0 = 1	
		const uint fermat_pass = eq96(fermat_mont.r0, one.r0);

		if( fermat_pass ){
			// use the PRP count as the "prime count"
			atomic_inc(&fermat_count);

			// subtract one to get the quotient = r1 of 2^(p - 1) - 1 mod p^2
			const cl_uint96_t quotient_mont = sub_mod96(fermat_mont.r1, one.r1, p);

			const uint pmod8 = p.lo & 7U;
			const int euler_sign = ((pmod8 == 3U) || (pmod8 == 5U)) ? -1 : 1;

			if( is_zero96(quotient_mont) ){
				// we have a Wieferich prime!
				store_result(p, euler_sign, 0, g_primecount, g_result);
			}
			else{
				// convert the quotient out of Montgomery form for calculating near misses
				const cl_uint96_t quotient = mont_get_modp96(quotient_mont, p, q);

				// use the full quotient for the checksum
				sum[lid] = quotient;

				// use the half-exponent quotient for "near-Wieferich" relative to the Euler sign
				// distance of 2^((p-1)/2) from +-1 mod p^2
				cl_uint96_t half_quotient = half_mod96(quotient, p);
				if(euler_sign < 0) half_quotient = sub_mod96(zero96(), half_quotient, p);

				const cl_uint96_t threshold = make96_3(SPECIAL_THRESHOLD, 0U, 0U);
				const cl_uint96_t delta = sub96_wrap(p, half_quotient);

				const uint positive_pass = le96(half_quotient, threshold);
				const uint negative_pass = le96(delta, threshold);

				if(positive_pass || negative_pass){
					int special;
					if(negative_pass && (!positive_pass || lt96(delta, half_quotient))){
						special = -small96_to_int(delta);
					}
					else{
						special = small96_to_int(half_quotient);
					}
					store_result(p, euler_sign, special, g_primecount, g_result);
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














