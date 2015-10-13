#include <stdio.h> // for stdin
#include <stdlib.h>
#include <unistd.h> // for ssize_t

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_OPENMP
#include <omp.h>
#endif


/* Function: Greatest Common Divisor */
int
gcd ( int a, int b )
{
	int c;
	while ( a != 0 ) {
		c = a; a = b%a;  b = c;
	}
	return b;
}


int main(int argc, char **argv) {
	char *inputLine = NULL; size_t lineLength = 0;
	int *start, *stop, *numThreads, amountOfRuns = 0;
	int myRank = 0, totalRanks = 1;


#ifdef HAVE_MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalRanks);
#endif

	if(myRank == 0)
	{
		// Read in first line of input
		getline(&inputLine, &lineLength, stdin);
		sscanf(inputLine, "%d", &amountOfRuns);

		stop = (int*) calloc(amountOfRuns, sizeof(int));
		start = (int*) calloc(amountOfRuns, sizeof(int));
		numThreads = (int*) calloc(amountOfRuns, sizeof(int));

		int tot_threads, current_start, current_stop;
		for (int i = 0; i < amountOfRuns; ++i){

			// Read in each line of input that follows after first line
			free(inputLine); lineLength = 0; inputLine = NULL;
			ssize_t readChars = getline(&inputLine, &lineLength, stdin);

			// If there exists at least two matches (2x %d)...
			int matches =sscanf(inputLine, "%d %d %d", &current_start, &current_stop, &tot_threads);
			if (matches >= 2){
				if(current_start < 0 || current_stop < 0 || current_stop <= current_start){
					current_start = 0, current_stop = 0;
				}
				stop[i] = current_stop;
				start[i] = current_start;
				numThreads[i] = 1;
			}
			if(matches >= 3)
			{
				numThreads[i] = tot_threads;
			}
			if(matches < 2)
			{
				stop[i] = 0;
				start[i] = 0;
			}
		}
	}
#ifdef HAVE_MPI
	MPI_Bcast (&amountOfRuns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if( myRank )
	{
		stop = (int*) calloc(amountOfRuns, sizeof(int));
		start = (int*) calloc(amountOfRuns, sizeof(int));
		numThreads = (int*) calloc(amountOfRuns, sizeof(int));
	}
	MPI_Bcast (start, amountOfRuns, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (stop, amountOfRuns, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (numThreads, amountOfRuns, MPI_INT,    0, MPI_COMM_WORLD);
#endif



	//globalSum is the sum across MPI ranks
	int globalSum;
	int c;


	for(int i = 0; i < amountOfRuns; i++)
	{
		globalSum = 0;
		if(myRank == 0)
		{
			if(stop[i] == 0)
			{
				printf("%d\n", globalSum);
				continue;
			}
		}

		for(int m = 2 + myRank; m < stop[i]; m += totalRanks)
		{
			//localSum is the sum across OMP threads
			int localSum = 0;

#pragma 	omp parallel for shared(localSum) num_threads(numThreads[i]) reduction( +: localSum)
			for(int n = 1; n < m; n++)
			{
				//innerSum is the sum within each loop iteration done by a thread
				int innerSum = 0;
				if(gcd(m, n) == 1 && ((m - n) & 0x1))
				{
					c = m * m + n * n;
					if(c >= start[i] && c < stop[i])
					{
						innerSum++;
					}
				}
#ifdef HAVE_OMP
				localSum = innerSum;
#else
				localSum += innerSum;
#endif
			}
			globalSum += localSum;
		}
#ifdef HAVE_MPI
		MPI_Reduce(myRank ? &globalSum : MPI_IN_PLACE, &globalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

		if(myRank == 0)
		{
			printf("%d\n", globalSum);
		}
	}

#ifdef HAVE_MPI
	MPI_Finalize();
#endif

	return 0;
}
