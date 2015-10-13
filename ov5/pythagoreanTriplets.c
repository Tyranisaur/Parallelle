#define _GNU_SOURCE
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

#ifdef HAVE_MPI
	MPI_Init(NULL, NULL);
#endif

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

	/*
	 *	Remember to only print 1 (one) sum per start/stop.
	 *	In other words, a total of <amountOfRuns> sums/printfs.
	 */
	int globalSum;
	int c;
	for(int i = 0; i < amountOfRuns; i++)
	{
		globalSum = 0;
		if(stop[i] == 0)
		{
			printf("%d\n", globalSum);
			continue;
		}

		for(int m = 2; m < stop[i]; m++)
		{
			int localSum = 0;

#pragma 	omp parallel for shared(localSum) num_threads(numThreads[i])
			for(int n = 1; n < m; n++)
			{
				int innerSum = 0;
				if(gcd(m, n) == 1 && ((m - n) & 0x1))
				{
					c = m * m + n * n;
					if(c >= start[i] && c < stop[i])
					{
						innerSum++;
					}
				}
#pragma 		omp critical
				localSum += innerSum;
			}
			globalSum += localSum;
		}




		printf("%d\n", globalSum);
	}
	return 0;
}
