CC = g++
LD = $(CC)

.SUFFIXES:
.SUFFIXES: .o .c .h .cl .cpp

VERSION_MAJOR := 0
VERSION_MINOR := 1
date := $(shell powershell.exe get-date -format FileDate)

APP = WieferichCL-win64-v$(VERSION_MAJOR).$(VERSION_MINOR)-$(date).exe

SRC = main.cpp cl_wieferich.cpp cl_wieferich.h simpleCL.c simpleCL.h kernels/clearresult.cl kernels/getsegprimes.cl kernels/clearn.cl kernels/wieferich.cl kernels/common.cl
KERNEL_HEADERS = kernels/clearresult.h kernels/getsegprimes.h kernels/clearn.h kernels/wieferich.h kernels/common.h
OBJ = main.o cl_wieferich.o simpleCL.o

LIBS = OpenCL.dll libgmpwin.a libpariwin.a

BOINC_DIR = C:/mingwbuilds/boinc
BOINC_INC = -I$(BOINC_DIR)/lib -I$(BOINC_DIR)/api -I$(BOINC_DIR) -I$(BOINC_DIR)/win_build
BOINC_LIB = -L$(BOINC_DIR)/lib -L$(BOINC_DIR)/api -L$(BOINC_DIR) -lboinc_opencl -lboinc_api -lboinc

CFLAGS  = -I . -I kernels -I C:\msys64\usr\bin\include\pari -O3 -m64 -Wall -DVERSION_MAJOR=\"$(VERSION_MAJOR)\" -DVERSION_MINOR=\"$(VERSION_MINOR)\"
LDFLAGS = $(CFLAGS) -lstdc++ -static

all : clean $(APP)

$(APP) : $(OBJ)
	$(LD) $(LDFLAGS) $^ $(LIBS) $(OCL_LIB) $(BOINC_LIB) -o $@

main.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ main.cpp

cl_wieferich.o : $(SRC) $(KERNEL_HEADERS)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ cl_wieferich.cpp

simpleCL.o : $(SRC)
	$(CC) $(CFLAGS) $(OCL_INC) $(BOINC_INC) -c -o $@ simpleCL.c

.cl.h:
	perl cltoh.pl $< > $@

clean :
	del *.o
	del kernels\*.h
	del $(APP)

