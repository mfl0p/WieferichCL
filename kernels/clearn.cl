/*

	clearn kernel

	Bryan Little April 2020

	Clears prime counter.

*/


__kernel void clearn(__global uint *g_primecount){

	int gid = get_global_id(0);

	if(!gid){
		g_primecount[0]=0;
	}


}



