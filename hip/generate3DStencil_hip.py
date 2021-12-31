#Copyright 2021 Matthias Knorr, https://github.com/MKKnorr, MKKnorr@web.de
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys

def err(message):
    print(message, file=sys.stderr)
    sys.exit(1)

usage = "generate3DStencil_hip.py usage:\n"
usage += "\tstencil neighbourhood\t\t'Box' or 'Star'\n"
usage += "\taccess orientation\t'Row' or 'Column'\n"
usage += "\tcoefficients\t'Jacobi' or 'Variable'\n"
usage += "\trange\n"
usage += "\tcoarsening-factor in x\n"
usage += "\tcoarsening-factor in y\n"
usage += "\tcoarsening-factor in z\n"
usage += "x-/y-/z coarsening-factors [>0] are elements processed by a single thread\n"

usage += "\n\toptional:\n"
usage += "\talignment\t\t'Write' or 'Read', default is 'None'\n"
usage += "\tsize of alignment (in units of grid-points)\n"

if(len(sys.argv) != 10 and len(sys.argv) != 8):
    err("Unknown arguments!\n" + usage)

neighbourhood = str(sys.argv[1])
orientation = str(sys.argv[2])
coefftype = str(sys.argv[3])
stencilRange = int(sys.argv[4])
xCoarsening = int(sys.argv[5])
yCoarsening = int(sys.argv[6])
zCoarsening = int(sys.argv[7])

alignment = "None"
CLsize = 0
if(len(sys.argv) == 10):
    alignment = str(sys.argv[8])
    CLsize = int(sys.argv[9])
    if(CLsize <= 0 and alignment != "None"):
        err("size of alignment has to be > 0\n" + usage)

xwidth = 2*stencilRange + xCoarsening
ywidth = 2*stencilRange + yCoarsening
#width is needed for indexing of registers, similar to grid indexing

if(neighbourhood != "Box" and neighbourhood != "Star"):
    err("Unknown stencil neighbourhood!\n" + usage)
if(orientation != "Row" and orientation != "Column"):
    err("Unknown access orientation!\n" + usage)
if(coefftype != "Jacobi" and coefftype != "Variable"):
    err("Unknown coefficient type!\n" + usage)
if(stencilRange < 1):
    err("Range has to be at least 1!\n" + usage)
if(xCoarsening < 1 or yCoarsening < 1 or zCoarsening < 1):
    err("x-/y- or z-coarsening factors have to be at least 1!\n" + usage)
if(alignment != "None" and alignment != "Write" and alignment != "Read"):
    err("Unkown alignment! Default is 'None'.\n" + usage)


stencilSize = 0
if(neighbourhood == "Box"):
    stencilSize = (2*stencilRange + 1)**3
if(neighbourhood == "Star"):
    stencilSize = 6*stencilRange + 1


def printFunctionHeader():
    functionHeader = "__global__ void updateGrid("
    functionHeader += "const unsigned long xdim, const unsigned long ydim, const unsigned long zdim, double * const __restrict__ array0, double * const __restrict__ array1"
    if(alignment == "Write" or alignment == "Read"):
        functionHeader += ", const unsigned int rightOffset"
    functionHeader += "){\n"
    return functionHeader

def printFunctionInit():
    functionInit = "\tconst size_t x = (blockIdx.x * blockDim.x + threadIdx.x)"
    if(xCoarsening != 1):
        functionInit += "*{0}".format(xCoarsening)
    functionInit += " + range"
    if(alignment == "Write" and (stencilRange%CLsize != 0)):
        leftOffset = (CLsize - stencilRange%CLsize)
        functionInit += " + {0}".format(leftOffset)
    functionInit += ";\n"
    functionInit += "\tconst size_t y = (blockIdx.y * blockDim.y + threadIdx.y)"
    if(yCoarsening != 1):
        functionInit += "*{0}".format(yCoarsening)
    functionInit += " + range;\n"
    functionInit += "\tconst size_t z = (blockIdx.z * blockDim.z + threadIdx.z)"
    if(zCoarsening != 1):
        functionInit += "*{0}".format(zCoarsening)
    functionInit += " + range;\n"
    return functionInit

def printCheck():
    check = "\tif((x"
    if((xCoarsening - 1) != 0):
        check += " + {0}".format(xCoarsening - 1)
    check += ") < (xdim - range"
    if(alignment != "None"):
        check += " - rightOffset"
    check += ") && (y"
    if((yCoarsening - 1) != 0):
        check += " + {0}".format(yCoarsening - 1)
    check += ") < (ydim - range) && (z"
    if((zCoarsening - 1) != 0):
        check += " + {0}".format(zCoarsening - 1)
    check += ") < (zdim - range)){\n"
    return check

def printRegLoad(x, y, z):
    signx = "+"
    signy = "+"
    signz = "+"
    if(x < 0):
        signx = "-"
    if(y < 0):
        signy = "-"
    if(z <  0):
        signz = "-"
    #registers are addressed relative as a block around origin-point (x,y,z), with most upper-left-behind point being (0,0,0)
    registerAddress = (x + stencilRange) + (y + stencilRange)*xwidth + (z + stencilRange)*xwidth*ywidth
    regLoad = "\t\tdouble register{0} = array0[(x {1} {2}) + (y {3} {4})*xdim + (z {5} {6})*xdim*ydim];\n".format(registerAddress, signx, abs(x), signy, abs(y), signz, abs(z))
    return regLoad

#calculate linear index for mask-array from relative coordinates
def calcMaskAddress(xGrid, yGrid, zGrid, xReg, yReg, zReg):
    #relative coordinates as seen from updated element
    xRel = xGrid - xReg
    yRel = yGrid - yReg
    zRel = zGrid - zReg
    if(neighbourhood == "Box"):
        xMask = xRel + stencilRange
        yMask = yRel + stencilRange
        zMask = zRel + stencilRange
        maskAddress = xMask + yMask*(2*stencilRange + 1) + zMask*(2*stencilRange + 1)**2
        return maskAddress
    if(neighbourhood == "Star"):
        if(zRel < 0):
            zMask = zRel + stencilRange
            maskAddress = zMask
            return maskAddress

        if(zRel == 0 and xRel == 0 and yRel < 0):
            offset = stencilRange #amount of elements in zRel < 0
            yMask = yRel + stencilRange
            maskAddress = yMask + offset
            return maskAddress
        if(zRel == 0 and yRel == 0):
            offset = 2*stencilRange
            xMask = xRel + stencilRange
            maskAddress = xMask + offset
            return maskAddress
        if(zRel == 0 and xRel == 0 and yRel > 0):
            offset = 3*stencilRange
            yMask = yRel + stencilRange
            maskAddress = yMask + offset
            return maskAddress

        if(zRel > 0):
            offset = 4*stencilRange
            zMask = zRel + stencilRange
            maskAddress = zMask + offset
            return maskAddress

def printInit(x, y, z, xResReg, yResReg, zResReg):
    #x/y/z are coordinates relative to thread-origin
    #x/y/zResReg are coordinates of result-variable relative to thread-origin
    calc = ""
    calc += "\t\tdouble result{0} = register{1}".format(xResReg + yResReg*xCoarsening + zResReg*xCoarsening*yCoarsening, x + stencilRange + (y + stencilRange)*xwidth + (z + stencilRange)*xwidth*ywidth)
    if(coefftype == "Variable"):
        maskAddress = calcMaskAddress(x, y, z, xResReg, yResReg, zResReg)
        calc += "*d_mask[{0}]".format(maskAddress)
    calc += ";\n"
    return calc

def printSingleAddition(x, y, z, xResReg, yResReg, zResReg):
    #x/y/z are coordinates relative to thread-origin
    #x/y/zResReg are coordinates of result-variable relative to thread-origin
    calc = "\t\tresult{0} += register{1}".format(xResReg + yResReg*xCoarsening + zResReg*xCoarsening*yCoarsening, x + stencilRange + (y + stencilRange)*xwidth + (z + stencilRange)*xwidth*ywidth)
    if(coefftype == "Variable"):
        maskAddress = calcMaskAddress(x, y, z, xResReg, yResReg, zResReg)
        calc += "*d_mask[{0}]".format(maskAddress)
    calc += ";\n"
    return calc

#prints the additions for a value loaded from (xGrid, yGrid, zGrid) for all concerning result-variables
def printAdditions(xGrid, yGrid, zGrid):
    #x/y/zGrid is relative grid coordinate as seen from thread ranging from (-stencilRange) to (stencilRange + x/y/zCoarsening)
    #x/y/zelem are coordinates of result-variables ranging from (0,0,0) to (xCoarsening-1, yCoarsening-1, zCoarsening-1)
    content = ""
    for zelem in range(0, zCoarsening, 1):
        for yelem in range(0, yCoarsening, 1):
            for xelem in range(0, xCoarsening, 1):

                if(neighbourhood == "Box"):
                    #if chebyshev distance/maximum metric to elem <= stencilRange: add to result register
                    if((abs(xGrid - xelem) <= stencilRange) and (abs(yGrid - yelem) <= stencilRange) and (abs(zGrid - zelem) <= stencilRange)):
                        #if grid-point is 'first' that is added to this elem don't add but initialize
                        if(xelem - stencilRange == xGrid and yelem - stencilRange == yGrid and zelem - stencilRange == zGrid):
                            content += printInit(xGrid, yGrid, zGrid, xelem, yelem, zelem)
                        else:
                            content += printSingleAddition(xGrid, yGrid, zGrid, xelem, yelem, zelem)

                if(neighbourhood == "Star"):
                    #if point is on same x/y-Row and distance <= stencilRange: add to result register
                    if((abs(xGrid - xelem) <= stencilRange and yelem == yGrid and zelem == zGrid) or ((abs(yGrid - yelem) <= stencilRange) and xelem == xGrid and zelem == zGrid) or ((abs(zGrid - zelem) <= stencilRange) and xelem == xGrid and yelem == yGrid)):
                        #no difference in initialization between 'Row' and 'Column' access orientation as opposed to 2D-Star
                        if(xelem == xGrid and yelem == yGrid and zelem - stencilRange == zGrid):
                            content += printInit(xGrid, yGrid, zGrid, xelem, yelem, zelem)
                        else:
                            content += printSingleAddition(xGrid, yGrid, zGrid, xelem, yelem, zelem)
    return content

def processElement(x, y, z):
    code = ""
    code += printRegLoad(x, y, z)
    code += printAdditions(x, y, z)
    return code

def writeBack(x, y, z):
    wb = "\t\tarray1[x + {0} + (y + {1})*xdim + (z + {2})*xdim*ydim] = result{3}".format(x, y, z, x + y*xCoarsening + z*xCoarsening*yCoarsening)
    if(coefftype == "Jacobi"):
        wb += "*c"
    wb += ";\n"
    return wb

#actual generation of code:
content = "//Copyright 2021 Matthias Knorr, https://github.com/MKKnorr, MKKnorr@web.de\n"
content += "//\n"
content += "//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n"
content += "//\n"
content += "//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n"
content += "//\n"
content += "//THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
content += "\n"
content += "#include <hip/hip_runtime.h>\n"
content += "//Stencil: {0}, range: {1}, xCoarsening: {2}, yCoarsening: {3}, zCoarsening: {4}, orientation: {5}, alignment: {6}, cacheline size: {7}\n".format(neighbourhood, stencilRange, xCoarsening, yCoarsening, zCoarsening, orientation, alignment, CLsize)
content += "#define range {0}\n".format(stencilRange)

if(coefftype == "Variable"):
    content += "\n__constant__ double d_mask[{0}];\n\n".format(stencilSize)

content += printFunctionHeader()
content += printFunctionInit()
content += printCheck()
    
#for each value in grid:
#   1. load value in variable('register')
#   2. add variable to all resultRegisters it needs to be added to
#write back results

content += "\t\t//grid is traversed in slices in z-direction from 'front' to 'back'\n"
content += "\t\t//slices are traversed according to access orientation:\n"
if(orientation == "Column"):
    content += "\t\t//column by column\n"
if(orientation == "Row"):
    content += "\t\t//row by row\n"
content += "\n"

if(neighbourhood == "Box"):
    if(orientation == "Column"):
        for z in range(-1*stencilRange, stencilRange + zCoarsening, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            for x in range(-1*stencilRange, stencilRange + xCoarsening, 1):
                content += "\t\t//column x = {0}\n".format(x)
                for y in range(-1*stencilRange, stencilRange + yCoarsening, 1):
                    content += processElement(x, y, z)
    if(orientation == "Row"):
        for z in range(-1*stencilRange, stencilRange + zCoarsening, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            for y in range(-1*stencilRange, stencilRange + yCoarsening, 1):
                content += "\t\t//row y = {0}\n".format(y)
                for x in range(-1*stencilRange, stencilRange + xCoarsening, 1):
                    content += processElement(x, y, z)

if(neighbourhood == "Star"):
    if(orientation == "Column"):
        content += "\t\t//in front of intersection\n"
        #line in z-direction in front of intersection
        for z in range(-1*stencilRange, 0, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            for x in range(0, xCoarsening, 1):
                content += "\t\t//column x = {0}\n".format(x)
                for y in range(0, yCoarsening, 1):
                    content += processElement(x, y, z)
            content += "\n"
        content += "\n\t\t//cross in the center/at the intersection\n"
        #cross in the center
        for z in range(0, zCoarsening, 1):
        content += "\t\t//slice z = {0}\n".format(z)
            #horizontal line left part without intersection
            for x in range(-1*stencilRange, 0, 1):
                content += "\t\t//column x = {0}\n".format(x)
                for y in range(0, yCoarsening, 1):
                    content += processElement(x, y, z)
            #vertical line in middle of stencil
            for x in range(0, xCoarsening, 1):
                content += "\t\t//column x = {0}\n".format(x)
                for y in range(-1*stencilRange, stencilRange + yCoarsening, 1):
                    content += processElement(x, y, z)
            #horizontal line right part without intersection
            for x in range(xCoarsening, stencilRange + xCoarsening, 1):
                content += "\t\t//column x = {0}\n".format(x)
                for y in range(0, yCoarsening, 1):
                    content += processElement(x, y, z)
            content += "\n"
        #line in z-direction behind intersection
        content += "\n\t\t//behind intersection\n"
        for z in range(zCoarsening, stencilRange + zCoarsening, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            for x in range(0, xCoarsening, 1):
                content += "\t\t//column x = {0}\n".format(x)
                for y in range(0, yCoarsening, 1):
                    content += processElement(x, y, z)
            content += "\n"

    if(orientation == "Row"):
        content += "\t\t//in front of intersection\n"
        #line in z-direction in front of intersection
        for z in range(-1*stencilRange, 0, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            for y in range(0, yCoarsening, 1):
                content += "\t\t//row y = {0}\n".format(y)
                for x in range(0, xCoarsening, 1):
                    content += processElement(x, y, z)
            content += "\n"
        content += "\n\t\t//cross in the center\n"
        #cross in the center
        for z in range(0, zCoarsening, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            #upper part
            for y in range(-1*stencilRange, 0, 1):
                content += "\t\t//row y = {0}\n".format(y)
                for x in range(0, xCoarsening, 1):
                    content += processElement(x, y, z)
            #line in the middle
            for y in range(0, yCoarsening, 1):
                content += "\t\t//row y = {0}\n".format(y)
                for x in range(-1*stencilRange, stencilRange + xCoarsening, 1):
                    content += processElement(x, y, z)
            #lower part
            for y in range(yCoarsening, stencilRange + yCoarsening, 1):
                content += "\t\t//row y = {0}\n".format(y)
                for x in range(0, xCoarsening, 1):
                    content += processElement(x, y, z)
            content += "\n"
        content += "\n\t\t//behind intersection\n"
        #line in z-direction behind intersection
        for z in range(zCoarsening, stencilRange + zCoarsening, 1):
            content += "\t\t//slice z = {0}\n".format(z)
            for y in range(0, yCoarsening, 1):
                content += "\t\t//row y = {0}\n".format(y)
                for x in range(0, xCoarsening, 1):
                    content += processElement(x, y, z)
            content += "\n"

content += "\n\t\t//write back\n"
if(coefftype == "Jacobi"):
    if(neighbourhood == "Box"):
        content += "\t\tconst double c = {0}; //1.0/((2*stencilRange + 1)**3)\n".format(1.0/((2*stencilRange + 1)**3))
        content += "\t\t//const double c = 1.0/{0}.0;\n".format((2*stencilRange + 1)**3)
    elif(neighbourhood == "Star"):
        content += "\t\tconst double c = {0}; //1.0/(6*stencilRange + 1)\n".format(1.0/(6*stencilRange + 1))
        content += "\t\t//const double c = 1.0/{0}.0;\n".format(6*stencilRange + 1)

if(orientation == "Column"):
    for z in range(0, zCoarsening, 1):
        for x in range(0, xCoarsening, 1):
            for y in range(0, yCoarsening, 1):
                content += writeBack(x, y, z)
if(orientation == "Row"):
    for z in range(0, zCoarsening, 1):
        for y in range(0, yCoarsening, 1):
            for x in range(0, xCoarsening, 1):
                content += writeBack(x, y, z)

content += "\t}\n"
content += "}\n"
content += "\n"

#init-function
content += "__global__ void init(double * array, const unsigned long xdim, const unsigned long ydim, const unsigned long zdim"
if(alignment == "Write" or alignment == "Read"):
    content += ", const unsigned int rightOffset"
content += "){\n"
content += "\tconst size_t x = blockIdx.x*blockDim.x + threadIdx.x"
if(alignment == "Write"):
    leftOffset = (CLsize - stencilRange%CLsize)
    if(stencilRange%CLsize == 0):
        leftOffset = 0
    content += " + {0}".format(leftOffset)
content += ";\n"
content += "\tconst size_t y = blockIdx.y*blockDim.y + threadIdx.y;\n"
content += "\tconst size_t z = blockIdx.z*blockDim.z + threadIdx.z;\n"
content += "\tif(x < xdim "
if(alignment == "Read" or alignment == "Write"):
    content += "- rightOffset "
content += "&& y < ydim && z < zdim){\n"
content += "\t\tconst size_t idx = x + y*xdim + z*xdim*ydim;\n"
content += "\t\tif(x "
if(alignment == "Write"):
    content += "- {0} ".format(leftOffset)
content += "< range){\n"
content += "\t\t\tif(idx%2 == 0){\n"
content += "\t\t\t\tarray[idx] = 0.75;\n"
content += "\t\t\t} else{\n"
content += "\t\t\t\tarray[idx] = 0.25;\n"
content += "\t\t\t}\n"
content += "\t\t} else if(y < range){\n"
content += "\t\t\tif(idx%2 == 0){\n"
content += "\t\t\t\tarray[idx] = 0.75;\n"
content += "\t\t\t} else{\n"
content += "\t\t\t\tarray[idx] = 0.25;\n"
content += "\t\t\t}\n"
content += "\t\t} else if(z < range){\n"
content += "\t\t\tif(idx%2 == 0){\n"
content += "\t\t\t\tarray[idx] = 0.75;\n"
content += "\t\t\t} else{\n"
content += "\t\t\t\tarray[idx] = 0.25;\n"
content += "\t\t\t}\n"
content += "\t\t} else{\n"
content += "\t\t\tarray[idx] = 0.0;\n"
content += "\t\t}\n"
content += "\t}\n"
content += "}"
print(content)
