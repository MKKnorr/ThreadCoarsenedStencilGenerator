#include <iostream>
#include <string>
#include "benchmark.h"

enum programMode{Undefined, Failure, Bench2D, Bench3D, Profile2D, Profile3D};

int main(int argc, char **argv){

	const int Bench2DArgc = 13;
	const int Profile2DArgc = Bench2DArgc + 2;

	//differentiate between same argc via dimension-parameter and then via programMode
	const int Bench3DArgc = Bench2DArgc + 2; //15 (additional zCoarsening and zGridSize as parameter)
	const int Profile3DArgc = Bench3DArgc + 3;

	std::string neighbourhood;
	unsigned int dimension; //convert from string to unsigned int
	unsigned int stencilRange;
	std::string coeffType;

	std::string orientation;
	std::string alignment;
	unsigned int CLsize;

	double maxTime;

	unsigned int xCoarsening, yCoarsening, zCoarsening;
	unsigned int xGridSize, yGridSize, zGridSize;

	//for run of a single threadblock size
	unsigned int xthreads, ythreads, zthreads;

	if(argc >= Bench2DArgc and argc <= Profile3DArgc){
		neighbourhood = argv[1];
		std::string strDimension = argv[2];
		stencilRange = atoi(argv[3]);
		coeffType = argv[4];
		orientation = argv[5];
		alignment = argv[6];
		CLsize = atoi(argv[7]);

		maxTime = atof(argv[8]);

		programMode inputType = Undefined;

		if(strDimension.compare("2D") == 0){
			dimension = 2;
		}
		else if(strDimension.compare("3D") == 0){
			dimension = 3;
		}
		else{
			std::cerr << "Unknown dimension: " << strDimension << std::endl;
			inputType = Failure;
		}

		if(inputType != Failure){
			xCoarsening = atoi(argv[9]);
			yCoarsening = atoi(argv[10]);

			if(dimension == 2 and argc == Bench2DArgc){
				xGridSize = atoi(argv[11]);
				yGridSize = atoi(argv[12]);
				inputType = Bench2D;
			}
			else if(dimension == 3 and argc == Bench3DArgc){
				zCoarsening = atoi(argv[11]);

				xGridSize = atoi(argv[12]);
				yGridSize = atoi(argv[13]);
				zGridSize = atoi(argv[14]);
				inputType = Bench3D;
			}
			else if(dimension == 2 and argc == Profile2DArgc){
				xthreads = atoi(argv[13]);
				ythreads = atoi(argv[14]);

				inputType = Profile2D;
			}
			else if(dimension == 3 and argc == Profile3DArgc){
				xthreads = atoi(argv[15]);
				ythreads = atoi(argv[16]);
				zthreads = atoi(argv[17]);

				inputType = Profile3D;
			}
			else{
				std::cerr << "Dimension and amount of arguments mismatch!" << std::endl;
				inputType = Failure;
			}
		}

		if(inputType != Failure){
			//sanity of input-arguments somewhat verified. Explicit check is done in stencilBenchmark-constructor
			stencilBenchmark benchmark;

			if(dimension == 2){
				benchmark = stencilBenchmark(neighbourhood, dimension, alignment, CLsize,
					coeffType, orientation, stencilRange,
					xCoarsening, yCoarsening,
					xGridSize, yGridSize,
					maxTime);
			}
			else if(dimension == 3){
				benchmark = stencilBenchmark(neighbourhood, dimension, alignment, CLsize,
							coeffType, orientation, stencilRange,
							xCoarsening, yCoarsening, zCoarsening,
							xGridSize, yGridSize, zGridSize,
							maxTime);
			}

			benchmark.initGPU();
			benchmark.initBenchmark();

			if(inputType == Bench2D or inputType == Bench3D){
				benchmark.run();
			}
			else if(inputType == Profile2D){
				benchmark.singleRun(xthreads, ythreads);
			}
			else if(inputType == Profile3D){
				benchmark.singleRun(xthreads, ythreads, zthreads);
			}

			benchmark.printResults();
			exit(EXIT_SUCCESS);
		}
	}

	std::cerr << "Unknown arguments, the relevant arguments have to match the generated kernel code." << std::endl;
	std::cerr << "Usage:" << std::endl;
	std::cerr << "\t\tstencil neighbourhood ('Box' or 'Star')" << std::endl;
	std::cerr << "\t\tdimension ('2D' or '3D')" << std::endl;
	std::cerr << "\t\tstencil range > 0" << std::endl;
	std::cerr << "\t\tcoefficients type ('Jacobi' or 'Variable')" << std::endl;
	std::cerr << "\t\torientation\t('Column' or 'Row')" << std::endl;
	std::cerr << "\t\talignment\t('None', 'Write' or 'Read')" << std::endl;
	std::cerr << "\t\tsize of cache line in units of grid elements" << std::endl;
	std::cerr << "\t\tmaximal runtime for single threadblock size in seconds" << std::endl;

	std::cerr << "\t\tcoarsening factors (x y z[only for 3D])" << std::endl << std::endl;
	std::cerr << "\t\tgrid size (x y z[only for 3D])" << std::endl << std::endl;

	std::cerr << "optional for run of single threadblock size:" << std::endl;
	std::cerr << "\t\tthread block size (x y z[only for 3D])" << std::endl;
	exit(EXIT_FAILURE);
}
