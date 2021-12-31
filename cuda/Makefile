
NVCC = nvcc

bench: main.cpp benchmark.o launchUpdateGrid.o primes.o
	$(NVCC) --compiler-options "-Wall" $^ -o $@

%.o: %.cpp
	$(NVCC) --compiler-options "-Wall" $^ -c -o $@

%.o: %.cu
	$(NVCC) --compiler-options "-Wall" $^ -c -o $@

clean:
	rm -f *.o
	rm -f bench
