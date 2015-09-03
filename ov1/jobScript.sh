#!/bin/bash

### PBS setups required for queuing system @ Vilje ###

#PBS -N MPIJob

### This is the account used by us in TDT4200 Fall 2015
#PBS -A ntnu605

### Resources we specify we need for this job.
#PBS -l select=2:ncpus=32:mpiprocs=16
#PBS -l walltime=00:04:00


cd $ { PBS_O_WORKDIR }
module load mpt/2.11
module load intelcomp/15.0.1
mpiexec -np 32 ./parallel 2 10000000