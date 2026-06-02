# WieferichCL

WieferichCL by Bryan Little and Yves Gallot

A BOINC enabled OpenCL Wieferich Prime search program.

## Requirements

* OpenCL v1.1
* 64 bit operating system

## How it works

1. Search parameters are given on the command line.
2. A small group of candidates are generated on the GPU with a mod 210 wheel and sieving with primes up to 2^16.
3. The group of candidates are tested with a combined Fermat base 2 PRP test and Wieferich test.
4. Repeat #2-3 until checkpoint. Gather and verify results from GPU. Report any results in the result file.
5. When search range is complete and there are no results "no results" will be printed in the result file.
6. A checksum will be printed at the end of the result file. It can be used to compare results in a BOINC quorum.
   

## Running the program
```
command line options
* -p #			Start p
* -P #			End P
* 			P range is 3 <= -p < -P < 2^96, [-p, -P) exclusive.
* -t #			Override default "near-Wieferich" threshold and use #.
			Only works when -p is greater than 2^64.  Threshold of 10000 is used when -p is below 2^64.
* -r filename		Override default result file (results-WW.txt) and use specified file name.
* -h			Print help

Program gets the OpenCL GPU device index from BOINC.  To run stand-alone, the program will
default to GPU 0 unless an init_data.xml is in the same directory with the format:

<app_init_data>
<gpu_type>NVIDIA</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>

or

<app_init_data>
<gpu_type>ATI</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>
```

## Related Links
* [Yves Gallot on GitHub](https://github.com/galloty)

