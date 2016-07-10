#
# Makefile CPU / CUDA
#

#-----------------------------------------------------------------------------

all: 	simple cpu global  

simple:	Makefile-simple
	make -f Makefile-simple

cpu:	Makefile-cpu
	make -f Makefile-cpu

global: Makefile-globalMem
	make -f Makefile-globalMem

clean:
	make -f Makefile-simple     clean
	make -f Makefile-cpu        clean
	make -f Makefile-globalMem  clean
