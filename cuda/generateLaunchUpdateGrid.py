#Copyright 2021 Matthias Knorr, https://github.com/MKKnorr, MKKnorr@web.de
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys

def err(string):
    print(string, file=sys.stderr)
    sys.exit(1)

usage = "generateLaunchUpdateGrid.py usage:\n"
usage += "\tstencil neighbourhood\t\teither 'Box' or 'Star'\n"
usage += "\tcoefficients\t\t'Jacobi' or 'Variable'(coefficients have to be set by host)\n"
usage += "\trange\n"
usage += "\tdimension\n"

usage += "\n\toptional:\n"
usage += "\talignment\t\t'Write' or 'Read'\n" #'Write' or 'Read'

if(len(sys.argv) != 6 and len(sys.argv) != 5):
    err("Unknown arguments!\n" + usage)

neighbourhood = str(sys.argv[1])
coefftype = str(sys.argv[2])
stencilRange = int(sys.argv[3])
dimension = str(sys.argv[4])

alignment = "None"
if(len(sys.argv) == 6):
    alignment = str(sys.argv[5])

if(neighbourhood != "Star" and neighbourhood != "Box"):
    err("Unknown stencil neighbourhood!\n" + usage)
if(coefftype != "Jacobi" and coefftype != "Variable"):
    err("Unknown coefficient-type!\n" + usage)
if(stencilRange < 1):
    err("Range has to be at least 1!\n" + usage)
if(dimension != "2D" and dimension != "3D"):
    err("Unknown dimension!\n" + usage)
if(alignment != "Write" and alignment != "Read" and alignment != "None"):
    err("Unknown alignment!\n" + usage)

content = "//Copyright 2021 Matthias Knorr, https://github.com/MKKnorr, MKKnorr@web.de\n"
content += "//\n"
content += "//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n"
content += "//\n"
content += "//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n"
content += "//\n"
content += "//THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
content += "\n"
content += "#include <chrono>\n"
content += "#include <iostream>\n"
content += "#include \"launchUpdateGrid.h\"\n"
content += "#include \"kernels.cu\"\n"
content += "\n"
content += "#define gpuErrchk(ans){gpuAssert((ans), __FILE__, __LINE__);}\n"
content += "\n"
content += "inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false){\n"
content += "\tif(code != cudaSuccess){\n"
content += "\t\tfprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);\n"
content += "\t\tif(abort){\n"
content += "\t\t\texit(code);\n"
content += "\t\t}\n"
content += "\t}\n"
content += "}\n"
content += "\n"
content += "void swap(double **array0, double **array1){\n"
content += "\tdouble *tmp = *array0;\n"
content += "\t*array0 = *array1;\n"
content += "\t*array1 = tmp;\n"
content += "}\n"
content += "\n"
content += "void copyMask(double* mask, unsigned int maskSize){\n"
if(coefftype == "Variable"):
    content += "\t//d_mask allocated in kernels.cu as __constant__\n"
    content += "\tgpuErrchk(cudaMemcpyToSymbol(d_mask, mask, sizeof(double)*maskSize));\n"
    content += "\tgpuErrchk(cudaDeviceSynchronize());\n"
content += "}\n"
content += "\n"
content += "int maxPotentialBlockSize(){\n"
content += "\tint minGridSize = 0;\n"
content += "\tint maxBlockSize = 0;\n"
content += "\tgpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, updateGrid))\n"
content += "\treturn maxBlockSize;\n"
content += "}\n"


def printUpdateGrid(dimension, alignment):
    call = "updateGrid<<<gridBlocks, blockSize>>>(gridSize.x, gridSize.y"
    if(dimension == "3D"):
        call += ", gridSize.z"
    call += ", d_array0, d_array1"
    if(alignment == "Write" or alignment == "Read"):
        call += ", rightOffset"
    call += ");"
    return call


content += "std::vector<double> benchmarkUpdateGrid(std::string coefftype, std::string neighbourhood, unsigned int stencilRange, unsigned int dimension, std::string alignment,\n"
content += "\t\tdouble maxTime, dim3 gridSize, dim3 blockSize, dim3 gridBlocks, double *d_array0, double *d_array1, unsigned int rightOffset){\n"

content += "\t//runtime sanity check, these parameters are 'hardcoded' parameters in kernels.cu:\n"
content += "\tbool passedSanityCheck = true;\n"
content += "\tif(coefftype.compare(\"{0}\") != 0)\n\t\tpassedSanityCheck = false;\n".format(coefftype)
content += "\tif(neighbourhood.compare(\"{0}\") != 0)\n\t\tpassedSanityCheck = false;\n".format(neighbourhood)
content += "\tif(stencilRange != {0})\n\t\tpassedSanityCheck = false;\n".format(stencilRange)
content += "\tif(dimension != "
if(dimension == "2D"):
    content += "2"
if(dimension == "3D"):
    content += "3"
content += ")\n\t\tpassedSanityCheck = false;\n"
content += "\tif(alignment.compare(\"{0}\") != 0)\n\t\tpassedSanityCheck = false;\n".format(alignment)
content += "\tif(passedSanityCheck == false)\n"
content += "\t\tthrow(cudaErrorInvalidValue);\n\n"

content += "\tdouble runtime = 0.0;\n"
content += "\n"
content += "\tstd::vector<std::chrono::steady_clock::time_point> vecTimestamps;\n"
content += "\tvecTimestamps.reserve(1024); //preallocate memory to prevent reallocation during time logging - amount of saved timepoints is 2^n\n"
content += "\n"
content += "\tvecTimestamps.push_back(std::chrono::steady_clock::now());\n"
content += "\n"
content += "\tunsigned long long iterations = 1;\n"
content += "\tfor(iterations=1; runtime <= maxTime; iterations*=2){\n"
content += "\t\tfor(uint64_t i = 0; i < iterations; ++i){\n"
content += "\t\t\t"
content += printUpdateGrid(dimension, alignment)
content += "\n"
content += "\t\t\tgpuErrchk(cudaDeviceSynchronize());\n"
content += "\t\t\tswap(&d_array0, &d_array1);\n"
content += "\t\t\tvecTimestamps.push_back(std::chrono::steady_clock::now());\n"
content += "\t\t}\n"
content += "\t\truntime = std::chrono::duration<double> (vecTimestamps.back() - vecTimestamps.front()).count();\n"
content += "\t}\n"
content += "\titerations/=2;\n"
content += "\n"
content += "\tstd::vector<double> durations;\n"
content += "\tfor(auto it = vecTimestamps.begin(); it != (vecTimestamps.end() - 1); it++){\n"
content += "\t\tdurations.push_back(std::chrono::duration<double> (*(it + 1) - *it).count());\n"
content += "\t}\n"
content += "\treturn durations;\n"
content += "}\n"


content += "void launchInit(dim3 gridSize, dim3 blockSize, dim3 gridBlocks, double *d_array, unsigned int rightOffset){\n"
content += "\tinit<<<gridBlocks, blockSize>>>(d_array, gridSize.x, gridSize.y"
if(dimension == "3D"):
    content += ", gridSize.z"
if(alignment == "Write" or alignment == "Read"):
    content += ", rightOffset"
content += ");\n"
content += "\tgpuErrchk(cudaDeviceSynchronize());\n"
content += "}"
print(content)
