/*
	WieferichCL
	by Bryan Little, May 2026
	and Yves Gallot

	Required minimum OpenCL version is 1.1
	CL_TARGET_OPENCL_VERSION 110 in simpleCL.h

	Search limits:  11 <= P < 2^96
*/

#include <unistd.h>
#include <getopt.h>
#include <cinttypes>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"
#include "cl_wieferich.h"

void help()
{
	printf("Welcome to WieferichCL, an OpenCL Wieferich prime finder.\n");
	printf("Program usage:\n");
	printf("-p #		Starting prime p\n");
	printf("-P #		End prime P\n");
	printf("		P range is 11 <= -p < -P < 2^96, [-p, -P) exclusive\n");
	printf("-t #		Override default \"near-Wieferich\" threshold (%u) and use #\n", SPECIAL_THRESHOLD_MAX);
	printf("		Threshold is limited to <= 10000 when -p is below 2^64\n");
	printf("-r filename	Override default result file and use 'filename'\n");
	printf("-h		Print this help\n");
	boinc_finish(EXIT_FAILURE);
}


/* Parse uint128 with optional 'e' exponent, returns 0=ok, -1=invalid, -2=out of range.
	Accepts either a full decimal integer or mantissa*10^exponent form, for example:
	20000000000000000000
	2e19

	This does not use strtoull() for the mantissa, so values larger than UINT64_MAX
	are accepted as long as the final value fits in the supplied __uint128_t range. */
int parse_uint128(__uint128_t *result, const char *str,	__uint128_t lo, __uint128_t hi)
{
	if (!result || !str || !*str) return -1;

	const char *p = str;
	__uint128_t num = 0;

	if (*p < '0' || *p > '9') return -1;

	while (*p >= '0' && *p <= '9') {
		uint32_t digit = (uint32_t)(*p - '0');

		if (num > hi / 10 || (num == hi / 10 && (__uint128_t)digit > hi % 10))
			return -2;

		num = num * 10 + digit;
		p++;
	}

	if (*p == 'e' || *p == 'E') {
		p++;
		if (*p < '0' || *p > '9') return -1;

		uint64_t exp = 0;
		while (*p >= '0' && *p <= '9') {
			uint32_t digit = (uint32_t)(*p - '0');

			if (exp > UINT64_MAX / 10 || (exp == UINT64_MAX / 10 && digit > UINT64_MAX % 10))
				return -2;

			exp = exp * 10 + digit;
			p++;
		}

		if (*p != '\0') return -1;

		while (num != 0 && exp-- > 0) {
			if (num > hi / 10) return -2;
			num *= 10;
		}
	}
	else if (*p != '\0') {
		return -1; // invalid character
	}

	if (num < lo || num > hi) return -2;

		*result = num;
	return 0;
}

/* Parse uint64 with optional 'e' exponent, returns 0=ok, -1=invalid, -2=out of range */
int parse_uint64(uint64_t *result, const char *str, uint64_t lo, uint64_t hi)
{
	__uint128_t tmp;
	int status = parse_uint128(&tmp, str, (__uint128_t)lo, (__uint128_t)hi);
	if (status == 0) *result = (uint64_t)tmp;
	return status;
}

/* parse_uint simply wraps parse_uint64 for 32-bit values */
int parse_uint(uint32_t *result, const char *str, uint32_t lo, uint32_t hi)
{
	uint64_t tmp;
	int status = parse_uint64(&tmp, str, lo, hi);
	if (status == 0) *result = (uint32_t)tmp;
	return status;
}

/* Full command line parser from string */
void parse_cmdline_string(const char *cmdline, workStatus *st, searchData *sd)
{
	if (!cmdline || !st || !sd) return;

	__uint128_t maxlo = ((__uint128_t)1 << 96)-3;
	__uint128_t maxhi = ((__uint128_t)1 << 96)-1;

	char buf[8192];
	strncpy(buf, cmdline, sizeof(buf)-1);
	buf[sizeof(buf)-1] = '\0';

	char *token = strtok(buf, " \t");
	while (token) {
		if (strcmp(token, "-p") == 0) {
			token = strtok(NULL, " \t");
			if (token && parse_uint128(&st->pmin, token, (__uint128_t)11, maxlo) != 0) {
				fprintf(stderr, "Invalid value for -p: %s\n", token);
				printf("\nInvalid value for -p: %s\n\n", token);
				boinc_finish(EXIT_FAILURE);
			}
		}
		else if (strcmp(token, "-P") == 0) {
			token = strtok(NULL, " \t");
			if (token && parse_uint128(&st->pmax, token, (__uint128_t)12, maxhi) != 0) {
				fprintf(stderr, "Invalid value for -P: %s\n", token);
				printf("\nInvalid value for -P: %s\n\n", token);
				boinc_finish(EXIT_FAILURE);
			}
		}
		else if (strcmp(token, "-r") == 0) {
			token = strtok(NULL, " \t");
			if (token) {
				sd->result_file = strdup(token);  // allocate a copy
				if (!sd->result_file) {
					fprintf(stderr, "Failed to allocate memory for result file\n");
					printf("\nFailed to allocate memory for result file\n\n");
					boinc_finish(EXIT_FAILURE);
				}
			}
		}
		else if (strcmp(token, "-t") == 0) {
			token = strtok(NULL, " \t");
			if (token && parse_uint(&st->threshold, token, 0, SPECIAL_THRESHOLD_MAX) != 0) {
				fprintf(stderr, "Invalid value for -t: %s\n", token);
				printf("\nInvalid value for -t: %s\n\n", token);
				boinc_finish(EXIT_FAILURE);
			}
		}
		else if (strcmp(token, "-s") == 0) {
			sd->test = true;
		}
		else if (strcmp(token, "-h") == 0) {
			help();
		}
		// unknown flag or extra token → ignore silently
		token = strtok(NULL, " \t");
	}
}

/* Join argc/argv into a single string with spaces */
char* join_argv(int argc, char *argv[])
{
	if (argc <= 0) return NULL;

	// Estimate required buffer size
	size_t total = 1;
	for (int i = 1; i < argc; i++)
		total += strlen(argv[i]) + 1; // +1 for space or null terminator

	char *cmdline = (char *)malloc(total);
	if (!cmdline) return NULL;

	cmdline[0] = '\0';

	for (int i = 1; i < argc; i++) {
		strcat(cmdline, argv[i]);
		if (i < argc - 1) strcat(cmdline, " ");
	}

	return cmdline; // must free later
}

#ifdef _WIN32
double getSysOpType()
{
    double ret = 0.0;
    NTSTATUS(WINAPI *RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        ret = (double)osInfo.dwMajorVersion;
    }
    return ret;
}
#endif


int main(int argc, char *argv[])
{
	sclHard hardware = {};
	searchData sd = {};
	sd.write_state_a_next = true;
	workStatus st = {};

	st.threshold = SPECIAL_THRESHOLD_MAX;
	sd.numresults = 1000000;

	// Initialize BOINC
	BOINC_OPTIONS options;
	boinc_options_defaults(options);
	options.normal_thread_priority = true;
	boinc_init_options(&options);

	fprintf(stderr, "\nWieferichCL v%s.%s by Bryan Little and Yves Gallot\n",VERSION_MAJOR,VERSION_MINOR);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	if(boinc_is_standalone()){
		printf("\nWieferichCL v%s.%s by Bryan Little and Yves Gallot\n",VERSION_MAJOR,VERSION_MINOR);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	}

	// Print out cmd line for diagnostics
	fprintf(stderr, "Command line: ");
	for (int i = 0; i < argc; i++)
		fprintf(stderr, "%s ", argv[i]);
	fprintf(stderr, "\n");

	char *cmdline = join_argv(argc, argv);
	if (!cmdline) {
		fprintf(stderr, "Failed to build command line string\n");
		return 1;
	}

	// hack to work around invalid args when running under app_info
	//	printf("%s\n",cmdline);
	parse_cmdline_string(cmdline, &st, &sd);
	free(cmdline);

	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context ctx;
	cl_command_queue queue;
	cl_int err = 0;

	int retval = 0;
	retval = boinc_get_opencl_ids(argc, argv, 0, &device, &platform);
	if (retval) {
		if(boinc_is_standalone()){
			printf("init_data.xml not found, using device 0.\n");

			err = clGetPlatformIDs(1, &platform, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetPlatformIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetPlatformIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetDeviceIDs() failed with %d\n", err );
				fprintf(stderr, "Error: clGetDeviceIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
		}
		else{
			fprintf(stderr, "Error: boinc_get_opencl_ids() failed with error %d\n", retval );
			exit(EXIT_FAILURE);
		}
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: clCreateContext() returned %d\n", err);
		exit(EXIT_FAILURE);
	}

	// OpenCL v2.0
	//cl_queue_properties qp[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	//queue = clCreateCommandQueueWithProperties(ctx, device, qp, &err);

	queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "Error: Creating Command Queue. (clCreateCommandQueueWithProperties) returned %d\n", err );
		exit(EXIT_FAILURE);
	}

	hardware.platform = platform;
	hardware.device = device;
	hardware.queue = queue;
	hardware.context = ctx;

	char device_name[1024];
	char device_vend[1024];
	char device_driver[1024];
	cl_uint CUs;
	cl_ulong LMS = 0;
	cl_ulong global_mem = 0;
	cl_ulong max_alloc = 0;

	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_vend), &device_vend, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_driver), &device_driver, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CUs, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &LMS, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}
	err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceInfo failed with %d\n", err );
		exit(EXIT_FAILURE);
	}

	fprintf(stderr, "GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n  Local Mem Size: \t%u bytes\n",
		device_name, device_vend, device_driver, CUs, (uint32_t)LMS);
	if(boinc_is_standalone()){
		printf("GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n  Local Mem Size: \t%u bytes\n",
			device_name, device_vend, device_driver, CUs, (uint32_t)LMS);
	}

	// check vendor and normalize compute units
	// kernel size will be determined by profiling so this doesn't have to be accurate.
	sd.computeunits = (uint32_t)CUs;
	char intel_s[] = "Intel";
	char arc_s[] = "Arc";
	char nvidia_s[] = "NVIDIA";

	if(strstr((char*)device_vend, (char*)nvidia_s) != NULL){
#ifdef _WIN32
		// pascal or newer gpu on windows 10,11 allows long kernel runtimes without screen refresh issues

		float winVer = (float)getSysOpType();

		if(winVer >= 10.0f){

			cl_uint ccmajor;
			err = clGetDeviceInfo(hardware.device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(ccmajor), &ccmajor, NULL);
			if ( err != CL_SUCCESS ) {
				printf( "Error checking device compute capability\n" );
				fprintf(stderr, "Error checking device compute capability\n");
				exit(EXIT_FAILURE);
			}

			if(ccmajor >= 6){
				sd.compute = true;
			}
		}

#else
		// linux
		// list of popular gpus without video output
		char dc0[] = "P100";
		char dc1[] = "V100";
		char dc2[] = "T4";
		char dc3[] = "A100";
		char dc4[] = "L4";
		char dc5[] = "H100";
		char dc6[] = "H200";
		char dc7[] = "B100";
		char dc8[] = "B200";

		if(	strstr((char*)device_name, (char*)dc0) != NULL
				|| strstr((char*)device_name, (char*)dc1) != NULL
			|| strstr((char*)device_name, (char*)dc2) != NULL
			|| strstr((char*)device_name, (char*)dc3) != NULL
			|| strstr((char*)device_name, (char*)dc4) != NULL
			|| strstr((char*)device_name, (char*)dc5) != NULL
			|| strstr((char*)device_name, (char*)dc6) != NULL
			|| strstr((char*)device_name, (char*)dc7) != NULL
			|| strstr((char*)device_name, (char*)dc8) != NULL){
			sd.compute = true;
		}

#endif
	}
	// Intel
	else if( strstr((char*)device_vend, (char*)intel_s) != NULL ){
		if( strstr((char*)device_name, (char*)arc_s) != NULL ){
			sd.computeunits /= 10;
		}
		else{
			sd.computeunits /= 20;
			fprintf(stderr,"Detected Intel integrated graphics\n");
		}
	}
	// AMD
	else{
		sd.computeunits /= 2;
	}

	if(!sd.computeunits) sd.computeunits++;

	if(sd.test){
		run_test(hardware, st, sd);
	}
	else{
		cl_wieferich(hardware, st, sd);
	}

	sclReleaseClHard(hardware);

	boinc_finish(EXIT_SUCCESS);

	return 0;
}

