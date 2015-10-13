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
			if(current_start < 0 || current_stop < 0){
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
		else
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
	int localSum;
	for(int i = 0; i < amountOfRuns; i++)
	{
		globalSum = 0;
		int a, b, c;
#pragma omp parallel for shared(globalSum) threadprivate(localSum) num_threads(numThreads[i])
		for(int m = 2; m < stop[i];m++)
		{
			localSum = 0;
			for(int n = 1; n < m; n++)
			{
				if(gcd(m, n) == 1 && ((m - n) & 0x1))
				{
					c = m * m + n * n;
					//printf(" m == %d\tn == %d\tc == %d\n", m, n, c);
					if(c >= start[i] && c < stop[i])
					{
						localSum++;
						//printf("added\n");
					}
					else if( c >= stop[i])
					{
						break;
					}
				}
			}
			globalSum += localSum;
		}




		printf("%d\n", globalSum);
	}
	return 0;
}
