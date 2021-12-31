
CC = hipcc

bench: main.o benchmark.o launchUpdateGrid.o primes.o
	$(CC) -Wall $^ -o $@

%.o: %.cpp
	$(CC) -Wall $^ -c -o $@

%.o: %.cu
	$(CC) -Wall $^ -c -o $@

clean:
	rm -f *.o
	rm -f bench
