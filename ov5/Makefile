.PHONY: clean all run mpi omp mpiomp serial run_serial run_omp run_big_serial run_big_omp

all: omp serial mpi mpiomp

serial: pythagoreanTriplets.c
	gcc -std=c99 -o serial_program pythagoreanTriplets.c
	
omp: pythagoreanTriplets.c
	gcc -std=c99 -o omp_program pythagoreanTriplets.c -fopenmp -DHAVE_OPENMP

mpi: pythagoreanTriplets.c
	mpicc -std=c99 -o mpi_program pythagoreanTriplets.c -DHAVE_MPI
	
mpiomp: pythagoreanTriplets.c
	mpicc -std=c99 -o combined_program pythagoreanTriplets.c -fopenmp -DHAVE_MPI -DHAVE_OPENMP
	
clean: rm -f serial omp_program mpi_program combined_program

run_serial: serial
	./serial_program < input.txt
	
run_omp: omp
	./omp_program < input.txt
	
run_big_serial: serial
	./serial_program < big_input.txt
	
run_big_omp: omp
	./omp_program < big_input.txt