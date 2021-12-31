#include "benchmark.h"

#include <iostream>
#include <cmath>
#include <sstream>
#include <tuple>
#include "primes.h"

#include "launchUpdateGrid.h"

stencilBenchmark::stencilBenchmark(){
	neighbourhood.clear();
	dimension = 0;
	stencilRange = 0;
	alignment.clear();
	CLsize = 0;
	orientation.clear();
	xCoarsening = 0;
	yCoarsening = 0;
	zCoarsening = 0;
	flopsPerUpdate = 0;

	maxTime = 0.0;
}

stencilBenchmark::stencilBenchmark(
	std::string _neighbourhood, unsigned int _dimension, std::string _alignment, unsigned int _CLsize,
	std::string _coeffType, std::string _orientation, unsigned int _stencilRange,
	unsigned int _xCoarsening, unsigned int _yCoarsening,
	unsigned int xGridSize, unsigned int yGridSize,
	double _maxTime)
	:
	neighbourhood(_neighbourhood), dimension(_dimension), alignment(_alignment), CLsize(_CLsize),
	coeffType(_coeffType), orientation(_orientation), stencilRange(_stencilRange),
	xCoarsening(_xCoarsening), yCoarsening(_yCoarsening),
	maxTime(_maxTime)
{
	gridSize = dim3(xGridSize, yGridSize);

	checkParameters();
	setTypeAndSize();
}

stencilBenchmark::stencilBenchmark(
	std::string _neighbourhood, unsigned int _dimension, std::string _alignment, unsigned int _CLsize,
	std::string _coeffType, std::string _orientation, unsigned int _stencilRange,
	unsigned int _xCoarsening, unsigned int _yCoarsening, unsigned int _zCoarsening,
	unsigned int xGridSize, unsigned int yGridSize, unsigned int zGridSize,
	double _maxTime)
	:
	neighbourhood(_neighbourhood), dimension(_dimension), alignment(_alignment), CLsize(_CLsize),
	coeffType(_coeffType), orientation(_orientation), stencilRange(_stencilRange),
	xCoarsening(_xCoarsening), yCoarsening(_yCoarsening), zCoarsening(_zCoarsening),
	maxTime(_maxTime)
{
	gridSize = dim3(xGridSize, yGridSize, zGridSize);

	checkParameters();
	setTypeAndSize();
}

void stencilBenchmark::setTypeAndSize(){

	if(neighbourhood.compare("Box") == 0){	//stencilSize = (2*stencilRange + 1) ** dimension;
		unsigned int stencilWidth = 2*stencilRange + 1;
		stencilSize = stencilWidth;
		for(unsigned int i = 0; i < dimension - 1; i++){
			stencilSize *= stencilWidth;
		}
	}
	else if(neighbourhood.compare("Star") == 0){
		stencilSize = 2*dimension*stencilRange + 1;
	}

	if(coeffType.compare("Jacobi") == 0){
		flopsPerUpdate = stencilSize; //N - 1 additions, 1 multiplication
	}
	else if(coeffType.compare("Variable") == 0){
		flopsPerUpdate = 2*stencilSize - 1; //N - 1 additions, N multiplications
	}
}


void stencilBenchmark::run(){

	if(dimension == 2){
		Benchmark2D();
	}
	else if(dimension == 3){
		Benchmark3D();
	}
}



void stencilBenchmark::initGPU(){

	int deviceCount;
	gpuErrchk(cudaGetDeviceCount(&deviceCount));
	if(deviceCount == 0){
		std::cerr << "No gpu found, exiting." << std::endl;
		exit(EXIT_FAILURE);
	}

	gpuErrchk(cudaSetDevice(0)); //just use first gpu
	cudaDeviceProp dProps;
	gpuErrchk(cudaGetDeviceProperties(&dProps, 0));

	//sets shared memory config for whole device - shared memory is not used
	gpuErrchk(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	maxThreads.x = dProps.maxThreadsDim[0];
	maxThreads.y = dProps.maxThreadsDim[1];
	maxThreads.z = dProps.maxThreadsDim[2];

	GPUname = dProps.name;
	warpSize = dProps.warpSize;
}

void stencilBenchmark::initBenchmark(){

	if(coeffType.compare("Variable") == 0){
		//generate and copy stencil mask to GPU
		double* mask = new double[stencilSize];
		for(unsigned int i = 0; i < stencilSize; i++){
			//test-mask is isotrope, result is same as for Jacobi-like stencils
			mask[i] = 1.0/stencilSize;
		}
		copyMask(mask, stencilSize);
		delete[] mask;
	}

	maxThreadsPerBlock = maxPotentialBlockSize(); //function from launchUpdateGrid.cu - returns max threads per block for the kernel, which can be smaller than dProps.maxThreadsPerBlock

	//gridSize: dimensions of the grid including border
	//gridElements: actually updated elements - grid without border
	//gridThreads: threads needed to calculate gridElements
	unsigned int xGrid = gridSize.x;
	unsigned int yGrid = gridSize.y;
	unsigned int zGrid = gridSize.z;

	//adapt geometry for specialized kernel - arbitrary grid dimensions need additional kernel for border
	//grid wihtout border has to be a multiple of the Coarsening factors in the corresponding dimension
	if(dimension == 2){
		gridSize = dim3(	xGrid - (xGrid - 2*stencilRange)%xCoarsening,
					yGrid - (yGrid - 2*stencilRange)%yCoarsening);
	
		gridElements = dim3(	gridSize.x - 2*stencilRange,
					gridSize.y - 2*stencilRange);
	
		gridThreads = dim3(	gridElements.x/xCoarsening,
					gridElements.y/yCoarsening);
	}
	else if(dimension == 3){
		gridSize = dim3(	xGrid - (xGrid - 2*stencilRange)%xCoarsening,
					yGrid - (yGrid - 2*stencilRange)%yCoarsening,
					zGrid - (zGrid - 2*stencilRange)%zCoarsening);
	
		gridElements = dim3(	gridSize.x - 2*stencilRange,
					gridSize.y - 2*stencilRange,
					gridSize.z - 2*stencilRange);
	
		gridThreads = dim3(	gridElements.x/xCoarsening,
					gridElements.y/yCoarsening,
					gridElements.z/zCoarsening);
	}

	//alignment
	{
		size_t xmem = gridSize.x;
		if(alignment.compare("Write") == 0){
			alignXwidthWrite(xmem, rightOffset);
		}
		else if(alignment.compare("Read") == 0){
			alignXwidthRead(xmem, rightOffset);
		}
	
		gridSize.x = xmem;
	}

	N = gridSize.x*gridSize.y;
	if(dimension == 3){
		N *= gridSize.z;
	}

	size_t freeGPUMem;
	size_t totalGPUMem;
	cudaMemGetInfo(&freeGPUMem, &totalGPUMem);

	if(2*sizeof(double)*N > freeGPUMem){
		std::cerr << "Requested grid size is too big for the GPU (needed/free/total: " << 2*sizeof(double)*N << " / " << freeGPUMem << " / " << totalGPUMem << ")" << std::endl;
		exit(EXIT_FAILURE);
	}

	gpuErrchk(cudaMalloc(&d_array0, sizeof(double)*N));
	gpuErrchk(cudaMalloc(&d_array1, sizeof(double)*N));

	dim3 initBlockSize;
	dim3 initGridBlocks;

	if(dimension == 2){
		//fixed init threads
		initBlockSize = dim3(32, 16);
		//init worksize must fill out whole grid, not only points that are updated later on
		initGridBlocks = dim3(	(gridSize.x + initBlockSize.x - 1)/initBlockSize.x,
					(gridSize.y + initBlockSize.y - 1)/initBlockSize.y);
	}
	else if(dimension == 3){
		//fixed init threads
		initBlockSize = dim3(16, 16, 4);
		//init worksize must fill out whole grid, not only points that are updated later on
		initGridBlocks = dim3(	(gridSize.x + initBlockSize.x - 1)/initBlockSize.x,
					(gridSize.y + initBlockSize.y - 1)/initBlockSize.y,
					(gridSize.z + initBlockSize.z - 1)/initBlockSize.z);
	}

	launchInit(gridSize, initBlockSize, initGridBlocks, d_array0, rightOffset);
	launchInit(gridSize, initBlockSize, initGridBlocks, d_array1, rightOffset);
	cudaDeviceSynchronize();
}


void stencilBenchmark::Benchmark2D(){

	unsigned int maxWarps = maxThreadsPerBlock/warpSize;
	for(unsigned int warps = 1; warps <= maxWarps; warps++){
		std::vector<std::pair<unsigned int, unsigned int>> combinations2D = primeCombinations2D(warps*warpSize);
		for(std::pair<unsigned int, unsigned int> threadCombination : combinations2D){
			unsigned int xthreads = threadCombination.first;
			unsigned int ythreads = threadCombination.second;

			dim3 blockSize(xthreads, ythreads);
			//pad dimensions to a multiple of the threadblock-size to be used as global-work-size
			dim3 gridBlocks((gridThreads.x + xthreads - 1)/xthreads, (gridThreads.y + ythreads - 1)/ythreads);

			if(xthreads > maxThreads.x or ythreads > maxThreads.y){
				continue;
			}

			try{
				std::vector<double> durations = benchmarkUpdateGrid(coeffType, neighbourhood, stencilRange, dimension, alignment, maxTime, gridSize, blockSize, gridBlocks, d_array0, d_array1, rightOffset);
				benchResults.push_back(std::make_pair(durations, blockSize));
			}
			catch(cudaError_t err){
				if(err != cudaSuccess){
					std::cerr << "kernel failed for blockSize: " << blockSize.x << "\t" << blockSize.y << "\t" << blockSize.z << std::endl;
				}
			}
		}
	}
}

void stencilBenchmark::singleRun(unsigned int xthreads, unsigned int ythreads){

	if(xthreads > maxThreads.x || ythreads > maxThreads.y){
		std::cerr << "Requested threadblock dimensions (" << xthreads << ", " << ythreads << ") bigger than maximal possible (" << maxThreads.x << ", " << maxThreads.y << "). Exiting." << std::endl;
		exit(EXIT_FAILURE);
	}
	if(xthreads == 0 || ythreads == 0){
		std::cerr << "One of the dimensions of the requested threadblock is 0 (" << xthreads << ", " << ythreads << "). Dimensions have to be at least 1. Exiting." << std::endl;
		exit(EXIT_FAILURE);
	}
	if(xthreads*ythreads > maxThreadsPerBlock){
		std::cerr << "Threadblock (" << xthreads << ", " << ythreads << ") with " << xthreads*ythreads << " threads is too big. Maximum threads per block are " << maxThreadsPerBlock << " threads for this kernel. Exiting." << std::endl;
		exit(EXIT_FAILURE);
	}

	dim3 blockSize(xthreads, ythreads);

	//pad dimensions to a multiple of the threadblock-size to be used as global worksize
	dim3 gridBlocks((gridThreads.x + xthreads - 1)/xthreads,
			(gridThreads.y + ythreads - 1)/ythreads);

	try{
		std::vector<double> durations = benchmarkUpdateGrid(coeffType, neighbourhood, stencilRange, dimension, alignment, maxTime, gridSize, blockSize, gridBlocks, d_array0, d_array1, rightOffset);
		benchResults.push_back(std::make_pair(durations, blockSize));
	}
	catch(cudaError_t err){
		if(err != cudaSuccess){
			std::cerr << "kernel failed for blockSize: " << blockSize.x << "\t" << blockSize.y << "\t" << blockSize.z << std::endl;
		}
	}
}



void stencilBenchmark::Benchmark3D(){

	unsigned int maxWarps = maxThreadsPerBlock/warpSize;
	for(unsigned int warps = 1; warps <= maxWarps; warps++){
		std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> combinations3D = primeCombinations3D(warps*warpSize);
		for(std::tuple<unsigned int, unsigned int, unsigned int> threadCombination : combinations3D){
			unsigned int xthreads = std::get<0>(threadCombination);
			unsigned int ythreads = std::get<1>(threadCombination);
			unsigned int zthreads = std::get<2>(threadCombination);

			if(xthreads > maxThreads.x or ythreads > maxThreads.y or zthreads > maxThreads.z){
				continue;
			}

			dim3 blockSize(xthreads, ythreads, zthreads);

			dim3 gridBlocks((gridThreads.x + blockSize.x - 1)/blockSize.x,
					(gridThreads.y + blockSize.y - 1)/blockSize.y,
					(gridThreads.z + blockSize.z - 1)/blockSize.z);

			try{
				std::vector<double> durations = benchmarkUpdateGrid(coeffType, neighbourhood, stencilRange, dimension, alignment, maxTime, gridSize, blockSize, gridBlocks, d_array0, d_array1, rightOffset);
				benchResults.push_back(std::make_pair(durations, blockSize));
			}
			catch(cudaError_t err){
				if(err != cudaSuccess){
					std::cerr << "kernel failed for blockSize: " << blockSize.x << "\t" << blockSize.y << "\t" << blockSize.z << std::endl;
				}
			}
		}
	}
}

void stencilBenchmark::singleRun(unsigned int xthreads, unsigned int ythreads, unsigned int zthreads){

	if(xthreads*ythreads*zthreads > maxThreadsPerBlock){
		std::cerr << "Threadblock (" << xthreads << ", " << ythreads << ", " << zthreads << ") with " << xthreads*ythreads*zthreads << " threads is too big. Maximum threads per block are " << maxThreadsPerBlock << " threads for this kernel. Exiting." << std::endl;
		exit(EXIT_FAILURE);
	}
	if(xthreads > maxThreads.x || ythreads > maxThreads.y || zthreads > maxThreads.z){
		std::cerr << "Requested threadblock dimensions (" << xthreads << ", " << ythreads << ", " << zthreads << ") bigger than maximal possible (" << maxThreads.x << ", " << maxThreads.y << ", " << maxThreads.z << "). Exiting." << std::endl;
		exit(EXIT_FAILURE);
	}
	if(xthreads == 0 || ythreads == 0 || zthreads == 0){
		std::cerr << "One of the dimensions of the requested threadblock is 0 (" << xthreads << ", " << ythreads << ", " << zthreads << "). Dimensions have to be at least 1. Exiting." << std::endl;
		exit(EXIT_FAILURE);
	}

	dim3 blockSize(xthreads, ythreads, zthreads);

	//pad dimensions to a multiple of the threadblock-size to be used as global worksize
	dim3 gridBlocks((gridThreads.x + blockSize.x - 1)/blockSize.x,
			(gridThreads.y + blockSize.y - 1)/blockSize.y,
			(gridThreads.z + blockSize.z - 1)/blockSize.z);

	try{
		std::vector<double> durations = benchmarkUpdateGrid(coeffType, neighbourhood, stencilRange, dimension, alignment, maxTime, gridSize, blockSize, gridBlocks, d_array0, d_array1, rightOffset);
		benchResults.push_back(std::make_pair(durations, blockSize));
	}
	catch(cudaError_t err){
		if(err != cudaSuccess){
			std::cerr << "kernel failed for blockSize: " << blockSize.x << "\t" << blockSize.y << "\t" << blockSize.z << std::endl;
		}
	}
}


void stencilBenchmark::printResults(){

	std::stringstream content;
	content << std::scientific;
	content << "#GPU: " << GPUname << "\n";
	content << "#grid size excluding border: x " << gridElements.x << " y " << gridElements.y;
	if(dimension == 3)
		content << " z " << gridElements.z;
	content << "\n";
	content << "#coarsening factors: x " << xCoarsening << " y " << yCoarsening;
	if(dimension == 3)
		content << " z " << zCoarsening;
	content << "\n";
	content << "#alignment: " << alignment << "\n";
	if(alignment.compare("None") != 0){
		content << "#aligned to " << CLsize*sizeof(double) << " byte\n";
	}
	content << "#orientation: " << orientation << "\n";
	content << "#coefficients: " << coeffType << "\n";
	content << "#neighbourhood: " << neighbourhood << "\n";
	content << "#range: " << stencilRange << "\n";
	content << "#dimension: " << dimension << "\n";

	content << "#xthreads\tythreads\tzthreads\tPerf [Flop/s]\tPerf dev [Flop/s]\tPerf [Lup/s]\tPerf dev [Lup/s]\titerations\ttotal time[s]\n";

	double updatedPoints = gridElements.x*gridElements.y;
	if(dimension == 3)
		updatedPoints *= gridElements.z;

	for(std::pair<std::vector<double>, dim3> result : benchResults){
		std::vector<double> vecRuntime = result.first;
		double meanFlops = 0.0;
		double meanLups = 0.0;
		double totalRuntime = 0.0;

		for(double runtime : vecRuntime){
			//lups = updated points/iteration
			double lups = updatedPoints/runtime;
			meanLups += lups;

			//flops = updated points/iteration * flops/update point
			double flops = lups*flopsPerUpdate;
			meanFlops += flops;

			totalRuntime += runtime;
		}
		meanFlops /= vecRuntime.size();
		meanLups /= vecRuntime.size();

		double stdDevLups = 0.0;
		for(double runtime: vecRuntime){
			//lups = updated points/iteration
			double lups = updatedPoints/runtime;

			stdDevLups += (lups - meanLups)*(lups - meanLups);
		}
		stdDevLups /= vecRuntime.size();
		stdDevLups = sqrt(stdDevLups);
		double stdDevFlops = flopsPerUpdate*stdDevLups;

		dim3 threads = result.second;

		content << threads.x << "\t" << threads.y << "\t";
		if(dimension == 3){
			content << threads.z;
		}
		else if(dimension == 2){
			content << "0";
		}
		
		content << "\t" << meanFlops << "\t" << stdDevFlops << "\t" << meanLups << "\t" << stdDevLups << "\t" << vecRuntime.size() << "\t" << totalRuntime << "\n";
	}

	std::cout << content.rdbuf();
}

void stencilBenchmark::checkParameters(){
	if((neighbourhood.compare("Box") != 0 and neighbourhood.compare("Star") != 0)){
		std::cerr << "Unknown stencil-neighbourhood: " << neighbourhood << std::endl;
		std::cerr << "Either 'Box' or 'Star'. Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(dimension != 2 and dimension != 3){
		std::cerr << "Unknown dimension: " << dimension << std::endl;
		std::cerr << "Either '2D' or '3D'. Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(alignment.compare("None") != 0 and alignment.compare("Write") != 0 and alignment.compare("Read") != 0){
		std::cerr << "Unknown alignment: " << alignment << std::endl;
		std::cerr << "Either 'None', 'Write' or 'Read'. Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(alignment.compare("None") != 0 and CLsize == 0){
		std::cerr << "size of alignment is 0, but alignment is given." << std::endl;
		std::cerr << "Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(coeffType.compare("Jacobi") != 0 and coeffType.compare("Variable") != 0){
		std::cerr << "Unknown coefficients: " << coeffType << std::endl;
		std::cerr << "Either 'Jacobi' or 'Variable'. Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(orientation.compare("Row") != 0 and orientation.compare("Column") != 0){
		std::cerr << "Unknown orientation: " << orientation << std::endl;
		std::cerr << "Either 'Row' or 'Column'. Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(stencilRange == 0){
		std::cerr << "Stencil range (" << stencilRange << ") must be greater than zero!" << std::endl;
		std::cerr << "Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(xCoarsening == 0 or yCoarsening == 0 or (dimension == 3 and zCoarsening == 0)){
		std::cerr << "Coarsening factors have to be greater than zero!" << std::endl;
		std::cerr << "x: " << xCoarsening << std::endl;
		std::cerr << "y: " << yCoarsening << std::endl;
		if(dimension == 3){
			std::cerr << "z: " << zCoarsening << std::endl;
		}
		std::cerr << "Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(gridSize.x == 0 or gridSize.y == 0 or (dimension == 3 and gridSize.z == 0)){
		std::cerr << "Grid dimensions have to be greater than zero!" << std::endl;
		std::cerr << "x: " << gridSize.x << std::endl;
		std::cerr << "y: " << gridSize.y << std::endl;
		if(dimension == 3){
			std::cerr << "z: " << gridSize.z << std::endl;
		}
		std::cerr << "Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
	if(maxTime < 0){
		std::cerr << "Time for iterations must be non-negative: " << maxTime << std::endl;
		std::cerr << "Exiting!" << std::endl;
		exit(EXIT_FAILURE);
	}
}
