#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <utility>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n",
                                cudaGetErrorString(code), file, line);
                if (abort)
                        exit(code);
        }
}

class stencilBenchmark{

	public:
	//3D constructor
	stencilBenchmark(
	std::string _neighbourhood, unsigned int _dimension, std::string _alignment, unsigned int CLsize,
	std::string _coeffType, std::string _orientation, unsigned int _stencilRange,
	unsigned int _xCoarsening, unsigned int _yCoarsening, unsigned int _zCoarsening,
	unsigned int xGridSize, unsigned int yGridSize, unsigned int zGridSize,
	double _maxTime);

	//2D constructor
	stencilBenchmark(
	std::string _neighbourhood, unsigned int _dimension, std::string _alignment, unsigned int CLsize,
	std::string _coeffType, std::string _orientation, unsigned int _stencilRange,
	unsigned int _xCoarsening, unsigned int _yCoarsening,
	unsigned int xGridSize, unsigned int yGridSize,
	double _maxTime);

	stencilBenchmark();

	//stencil-specifics
	std::string neighbourhood;
	unsigned int dimension;
	std::string alignment;
	unsigned int CLsize;
	std::string coeffType;
	std::string orientation;
	unsigned int stencilRange;
	unsigned int xCoarsening, yCoarsening, zCoarsening;


	unsigned int stencilSize;
	unsigned int flopsPerUpdate;


	//benchmark-specifics
	double maxTime;
	dim3 gridSize; //amount of elements including border
	std::vector<std::pair<std::vector<double>, dim3>> benchResults;

	//benchmark internals
	size_t xmem;
	unsigned int rightOffset;


	//gridElements is the amount of elements in the grid without border
	dim3 gridElements;
	//gridThreads is the actual amount of threads that would have to be running when gridThreads is a multiple of blockThreads
	dim3 gridThreads;
	size_t N; //total gridSize, including border, in elements

	double *d_array0;
	double *d_array1;

	//gpu/cuda-specifics
	std::string GPUname;
	unsigned int maxThreadsPerBlock;
	dim3 maxThreads;
	unsigned int warpSize;

	void run();//run complete benchmark - with gpuinit, benchmarkinit, prebench and benchmarks

	//for profiling or other purposes:
	void singleRun(unsigned int xthreads, unsigned int ythreads);
	void singleRun(unsigned int xthreads, unsigned int ythreads, unsigned int zthreads);

	void initGPU();

	void initBenchmark();

	void printResults();


	private:
	void setTypeAndSize();
	void checkParameters();

	void Benchmark2D();
	void Benchmark3D();

	//alignment helper function:
	void alignXwidthRead(size_t &xmem, unsigned int &rightOffset){
		//make xmem multiple of CLsize
		rightOffset = 0;
		if(xmem%CLsize != 0){ //align only if necessary
			rightOffset = CLsize - xmem%CLsize;
			xmem += rightOffset;
		}
	}

	//only right offset is needed as variable in kernel, even for left-aligned
	void alignXwidthWrite(size_t &xmem, unsigned int &rightOffset){
		unsigned int leftOffset = 0;
		if(stencilRange%CLsize != 0){
			//add to the left of the x-rows (CLsize - range%CLsize) elements to align the write accesses and somewhat the read accesses(at least for star)
			leftOffset = CLsize - stencilRange%CLsize;
		}
		xmem += leftOffset;
		rightOffset = 0;
		//make xmem multiple of CLsize
		if(xmem%CLsize != 0){
			rightOffset = CLsize - xmem%CLsize;
		}
		xmem += rightOffset;
	}
};
