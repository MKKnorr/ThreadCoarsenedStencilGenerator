#!/bin/bash
err() { >&2 echo "$@"; }

#Script to generate and benchmark a thread-coarsened stencil for given parameters.
#The stencil is tested on two squared/cubed grids (one read, one write) of roughly 1GB size each
#for all possible threadblock-sizes, each at least 1s

#gridsizes, two grids are allocated
x2DGridsize=11500
y2DGridsize=11500

x3DGridsize=500
y3DGridsize=500
z3DGridsize=500

#rough runtime for a single threadblock
time=1.0

Args2D=9
#Args2D+1
Args3D=10


if [ $# != ${Args2D} ] && [ $# != ${Args3D} ]; then
	err "Unknown arguments"
	err "Usage:"
	err "	neighbourhood			'Box' or 'Star'"
	err "	dimension			'2D' or '3D'"
	err "	stencilrange			>=1"
	err "	coefficients			'Jacobi' or 'Variable'"
	err "	orientation			'Row' or 'Column'"
	err "	alignment			'None', 'Write' or 'Read'"
	err "	size of alignment		in units of grid-points"
	err "	x/y/z-coarsening factors	>=1, value for z only needed in 3D"
	exit 1
fi

neighbourhood=${1}
dimension=${2}
range=${3}
coefficients=${4}
orientation=${5}
alignment=${6}
CLsize=${7}
xCoarsening=${8}
yCoarsening=${9}

#check argument sanity
#following checks also exit when argument is set to empty string
if [ ${dimension} == "2D" ] && [ $# == ${Args2D} ]; then
	:
elif [ ${dimension} == "3D" ] && [ $# == ${Args3D} ]; then
	zCoarsening=${10}
else
	err "Dimensions and amount of arguments mismatch"
fi

if [ ${neighbourhood} != "Box" ] && [ ${neighbourhood} != "Star" ]; then
	err "Unknown neighbourhood"
fi

if [ ${dimension} != "2D" ] && [ ${dimension} != "3D" ]; then
	err "Unknkown dimension"
fi

if ! [[ ${range} =~ ^[0-9]+$ ]]; then
	err "range doesn't seem to be an integer"
fi

if ! [[ ${range} -ge 1 ]]; then
	err "range has to be at least 1"
fi

if [ ${coefficients} != "Jacobi" ] && [ ${coefficients} != "Variable" ]; then
	err "Unknown coefficients"
fi

if [ ${orientation} != "Column" ] && [ ${orientation} != "Row" ]; then
	err "orientation must be 'Column' or 'Row'"
fi

if [ ${alignment} != "None" ] && [ ${alignment} != "Left" ] && [ ${alignment} != "Right" ]; then
	err "Unknown alignment"
fi

if ! [[ ${CLsize} =~ ^[0-9]+$ ]]; then
	err "alignment size doesn't seem to be an integer"
fi

if [ ${alignment} == "None" ] && [ ${CLsize} != 0 ]; then
	err "alignment size must be 0 if alignment is \"None\""
fi

if ! [[ ${xCoarsening} =~ ^[0-9]+$ ]]; then
	err "Coarsening factor in x-direction doesn't seem to be an integer"
elif ! [[ ${xCoarsening} -ge 1 ]]; then
	err "Coarsening factor in x-direction has to be at least 1"
fi

if ! [[ ${yCoarsening} =~ ^[0-9]+$ ]]; then
	err "Coarsening factor in y-direction doesn't seem to be an integer"
elif ! [[ ${yCoarsening} -ge 1 ]]; then
	err "Coarsening factor in y-direction has to be at least 1"
fi

if [ ${dimension} == "3D" ] && [ $# == ${Args3D} ]; then
	if ! [[ ${zCoarsening} =~ ^[0-9]+$ ]]; then
		err "Coarsening factor in z-direction doesn't seem to be an integer"
	elif ! [[ ${zCoarsening} -ge 1 ]]; then
		err "Coarsening factor in z-direction has to be at least 1"
	fi
fi

#actual script

make clean

python3 generateLaunchUpdateGrid.py ${neighbourhood} ${coefficients} ${range} ${dimension} ${alignment} > launchUpdateGrid.cu
if [ $? -ne 0 ]; then
	err "generateLaunchUpdateGrid.py failed, exiting"
fi

if [ ${dimension} == "2D" ]; then
	python3 generate2DStencil_hip.py ${neighbourhood} ${orientation} ${coefficients} ${range} ${xCoarsening} ${yCoarsening} ${alignment} ${CLsize} > kernels.cu
	if [ $? -ne 0 ]; then
		err "generate2DStencil_hip.py failed, exiting"
	fi

	make bench -j 4
	if [ $? -ne 0 ]; then
		err "make bench failed, exiting"
	fi

	./bench ${neighbourhood} ${dimension} ${range} ${coefficients} ${orientation} ${alignment} ${CLsize} ${time} ${xCoarsening} ${yCoarsening} ${x2DGridsize} ${y2DGridsize}
	if [ $? -ne 0 ]; then
		err "./bench ${neighbourhood} ${dimension} ${range} ${coefficients} ${orientation} ${alignment} ${CLsize} ${time} ${xCoarsening} ${yCoarsening} ${x2DGridsize} ${y2DGridsize} failed!"
	fi
elif [ ${dimension} == "3D" ]; then
	python3 generate3DStencil_hip.py ${neighbourhood} ${orientation} ${coefficients} ${range} ${xCoarsening} ${yCoarsening} ${zCoarsening} ${alignment} ${CLsize} > kernels.cu
	if [ $? -ne 0 ]; then
		err "generate3DStencil_hip.py bench failed, exiting"
	fi

	make bench -j 4
	if [ $? -ne 0 ]; then
		err "make bench failed, exiting"
	fi

	./bench ${neighbourhood} ${dimension} ${range} ${coefficients} ${orientation} ${alignment} ${CLsize} ${time} ${xCoarsening} ${yCoarsening} ${zCoarsening} ${x3DGridsize} ${y3DGridsize} ${z3DGridsize}
	if [ $? -ne 0 ]; then
		err "./bench ${neighbourhood} ${dimension} ${range} ${coefficients} ${orientation} ${alignment} ${CLsize} ${time} ${xCoarsening} ${yCoarsening} ${zCoarsening} ${x3DGridsize} ${y3DGridsize} ${z3DGridsize} failed!"
	fi
fi
