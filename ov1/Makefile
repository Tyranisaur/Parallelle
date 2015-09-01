.PHONY: clean all

START:=2
STOP:=10000000
PROCESSES:=

serial: computeSerial.c
	gcc -o $@ computeSerial.c -std=c99 -lm -O2
	
 
all: computeMPI.c
	mpicc -std=c99 -o parallel computeMPI.c 

run_serial: serial
	./serial $(START) $(STOP)

clean:
	rm -f serial
	rm -f better
