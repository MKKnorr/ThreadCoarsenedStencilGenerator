# ThreadCoarsenedStencilGenerator #

The core of this project are python-scripts (*generate{2,3}DStencil_{cuda,hip}.py*), that generate cuda or hip kernels for stencils of configurable range and neighbourhood with various optimizations, but the focus lies on thread-coarsening.\
Host-code to benchmark the performance of varying parameters for the generated kernels is also provided.

## Why thread-coarsening? ##
At a certain range for long-ranging stencils the L1-cache-bandwidth is the limiter for the performance on modern GPUs.\
On e.g. the AMD Radeon VII and MI100 or the NVIDIA V100 and A100, the L1-cache is designed such that for 1 Floating-Point-Instruction per 16 Byte (two double values) of L1-data, which is given for stencils, the maximal performance is 1/2 the peak-compute performance.\
By using thread-coarsening, the L1-balance (byte transferred from L1-cache per operation) can be reduced, s.t. the L1-bottleneck is overcome, since thread-coarsening enables the reuse of values in registers, which in turn enable peak-performance.

## Usage ##
Generally, running the scripts or binaries without parameters prints their usage.

The most simple way to get performance data is to use the provided **RunBenchmark.sh**-script in the cuda/hip-folders.

It generates the kernels with generate{2,3}DStencil_{cuda,hip}.py for a given set of parameters, and generates a general interface for the host to use these kernels with generateLaunchUpdateGrid.py. The interface-generation is needed in this case, since the parameters of the generated kernels can change, depending on the applied optimizations.\
By default this code is then run on grids of roughly 1GB size for all possible thread-block-dimensions for 1 second each, in order to find the best performing one.
The size of the grid and the runtime can be set in RunBenchmark.sh.\
The output lists the performance of each thread-block dimension in Flop/s and Lup/s (lattice-updates per second) with standard-deviation and the total runtime and iterations achieved in that time.

The benchmark can also be run for a single thread-block dimension by running "bench" with the appropriate parameters.

In order to manually compile the host-code without the RunBenchmark.sh-script, the output of **generate{2,3}DStencil_{cuda,hip}.py** has to be piped to **kernels.cu** and the output of **generateLaunchUpdateGrid.py** has to be piped to **launchUpdateGrid.cu**. The code is then compiled by calling **make bench**.

## Parameters ##
The code-generators provide several parameters, some of these change the used stencil, others are optimizations applied to those stencils:

- **dimension**\
The code-generators are available in 2D or 3D

- **stencil neighbourhood**\
Shape of the included elements in the stencil-calculation
*Box* or *Star*-shaped stencils

- **access orientation**\
Direction in which the accessed elements are traversed by a thread\
*Row* by row or *Column* by column\
Only significantly affects performance of GPUs with insufficient L1-cache-capacity\
In 3D the slices in z-direction are always traversed from "front to back\

- **stencil coefficients**\
The kind of coefficients the stencil uses for the weighting of the neighbouring elements\
*Jacobi* or *Variable*\
Jacobi: the elements in the neighbourhood of the updated element are summed up and averaged by a single multiplication\
Variable: the weighting for the individual elements in the neighbourhood can be specified by providing a mask. This mask is declared as \_\_constant\_\_, such that it is stored in the constant-cache on NVIDIA-GPUs or provided via the scalar-cache on AMD-GPUs, in this way the accesses to the mask don't thrash the L1-data-cache.

- **Range**\
Maximal distance from the updated-point to the neighbourhood-elements that are considered in the calculation.\
The way the distance is calculated depends on the chosen stencil-neighbourhood.\
For Box it is the chebyshev distance, for the Star stencil it is the normal distance between the points, but only points that lay on a common axis are included.

- **Coarsening-factors**\
Amount of points that are updated by a single thread, available in each dimension (*x y*) or (*x y z*).

- **Alignment**\
Flag for aligning the accesses to a certain boundary\
'*None*', '*Read*' or '*Write*'\
For 'Read' the accesses are aligned, such that the accessed elements of a thread-block start at a multiple of the given alignment size in x-direction.\
For 'Write' the accesses are aligned, such that the updated elements of a thread-block start at a multiple of the given alignment size in x-direction.\
The memory is layed out in such a way, that it is continous in x-direction.

- **Alignment size**\
Needed when alignment-flag is given. The alignment size is given in units of grid-points. The generated kernels use double-precision, so a grid-point is 8 byte.\
**When the alignment is the size of a cache line, the rows are padded in such a way, that the alignment is consistent relative to a cache line across all rows for the grid**, i.e. 'Read' always aligns the read-accesses of a thread-block to a cache line, and 'Write' always aligns the write-accesses of a thread-block to a cache line. Otherwise the alignment is not necessarily consistent across the rows.\
On the AMD Radeon VII e.g. a cache line is 64 byte (https://www.techpowerup.com/gpu-specs/docs/amd-vega-isa.pdf), or on modern NVIDIA GPUs the cache lines are of 128 byte size divided into 4x32 byte sectors (https://forums.developer.nvidia.com/t/pascal-l1-cache/49571/12).


## Limitations ##
The generated kernels can only work on grids that are a integer mulitiple of the coarsening-factor in the corresponding dimensions. For dimensions where the gridsize%coarsening-factor != 0, an extra handling for the border is needed. Since this border usually is negligible in comparison to the computational cost of the whole grid, this special handling was not implemented in these kernels, because they were only developed to show the potential of thread-coarsening for stencils.\
In the host-code this case is handled by truncating the grid to a multiple of the coarsening-factor.

The code has been tested with rocm 3.9.0 and cuda 11.2.
