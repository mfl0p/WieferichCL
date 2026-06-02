/*

	clearresult.cl - Bryan Little May 2026
	
	clear prime counters and checksum

*/

__kernel void clearresult(__global uint *g_primecount, __global ulong *g_total_pcnt, __global cl_uint96_t *g_cksum, const uint numgroups){

	uint gid = get_global_id(0);

	if(gid < numgroups){
		g_cksum[gid] = zero96();
		g_total_pcnt[gid] = 0;		// total primecount between checkpoints  
	}

	if(!gid){
		g_primecount[1] = 0;		// largest g_primecount[0] since checkpoint
		g_primecount[2] = 0;		// # of results 
	}

}

