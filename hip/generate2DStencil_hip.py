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

usage = "generate2DStencil_hip.py usage:\n"
usage += "\tstencil neighbourhood\t\t'Box' or 'Star'\n"
usage += "\taccess orientation\t'Row' or 'Column'\n"
usage += "\tcoefficients\t'Jacobi' or 'Variable'\n"
usage += "\trange\n"
usage += "\tcoarsening-factor in x\n"
usage += "\tcoarsening-factor in y\n"
usage += "x-/y coarsening-factors[>0] are elements processed by a single thread\n"

usage += "\n\toptional:\n"
usage += "\talignment\t\t'Write' or 'Read', default is 'None'\n"
usage += "\tsize of alignment (in units of grid-points)\n"

if(len(sys.argv) != 9 and len(sys.argv) != 7):
    err("Unknown arguments!\n" + usage)

neighbourhood = str(sys.argv[1])
orientation = str(sys.argv[2])
coefftype = str(sys.argv[3])
stencilRange = int(sys.argv[4])
xCoarsening = int(sys.argv[5])
yCoarsening = int(sys.argv[6])

alignment = "None"
CLsize = 0
if(len(sys.argv) == 9):
    alignment = str(sys.argv[7])
    CLsize = int(sys.argv[8])
    if(CLsize <= 0 and alignment != "None"):
        err("size of alignment has to be > 0\n" + usage)

xwidth = 2*stencilRange + xCoarsening
#width is needed for indexing of registers, similar to grid indexing

if(neighbourhood != "Box" and neighbourhood != "Star"):
    err("Unknown stencil neighbourhood!\n" + usage)
if(orientation != "Row" and orientation != "Column"):
    err("Unknown access orientation!\n" + usage)
if(coefftype != "Jacobi" and coefftype != "Variable"):
    err("Unknown coefficient type!\n" + usage)
if(stencilRange < 1):
    err("Range has to be at least 1!\n" + usage)
if(xCoarsening < 1 or yCoarsening < 1):
    err("x- or y-coarsening factors have to be at least 1!\n" + usage)
if(alignment != "None" and alignment != "Write" and alignment != "Read"):
    err("Unkown alignment! Default is 'None'.\n" + usage)


stencilSize = 0
if(neighbourhood == "Box"):
    stencilSize = (2*stencilRange + 1)**2
if(neighbourhood == "Star"):
    stencilSize = 4*stencilRange + 1


def printFunctionHeader():
    functionHeader = "__global__ void updateGrid("
    functionHeader += "const unsigned long xdim, const unsigned long ydim, const double * const __restrict__ array0, double * const __restrict__ array1"
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
    check += ") < (ydim - range)){\n"
    return check

#explicitly load value from array into variable
def printRegLoad(x, y):
    signx = "+"
    signy = "+"
    if(x < 0):
        signx = "-"
    if(y < 0):
        signy = "-"
    #registers are addressed relative as a block around origin-point (x,y,z), with most upper-left-behind point being (0,0,0)
    registerAddress = (x + stencilRange) + (y + stencilRange)*xwidth
    regLoad = "\t\tdouble register{0} = array0[(x {1} {2}) + (y {3} {4})*xdim];\n".format(registerAddress, signx, abs(x), signy, abs(y))
    return regLoad

#calculate linear index for mask-array from relative coordinates
def calcMaskAddress(xGrid, yGrid, xReg, yReg):
    #relative coordinates as seen from updated element
    xRel = xGrid - xReg
    yRel = yGrid - yReg
    if(neighbourhood == "Box"):
        xMask = xRel + stencilRange
        yMask = yRel + stencilRange
        maskAddress = xMask + yMask*(2*stencilRange + 1)
        return maskAddress
    if(neighbourhood == "Star"):
        if(xRel == 0 and yRel < 0):
            #upper part
            yMask = yRel + stencilRange
            maskAddress = yMask
            return maskAddress
        if(yRel == 0):
            #horizontal line
            xMask = xRel + stencilRange
            maskAddress = stencilRange + xMask
            return maskAddress
        if(xRel == 0 and yRel > 0):
            #lower part
            yMask = yRel + stencilRange
            maskAddress = yMask + 2*stencilRange #2*stencilRange from horizontal line
            return maskAddress

def printInit(x, y, xResReg, yResReg):
    #x/y are coordinates relative to thread-origin
    #x/yResReg are coordinates of result-variable
    calc = ""
    calc += "\t\tdouble result{0} = register{1}".format(xResReg + yResReg*xCoarsening, x + stencilRange + (y + stencilRange)*xwidth)
    if(coefftype == "Variable"):
        maskAddress = calcMaskAddress(x, y, xResReg, yResReg)
        calc += "*d_mask[{0}]".format(maskAddress)
    calc += ";\n"
    return calc

def printSingleAddition(x, y, xResReg, yResReg):
    #x/y are coordinates relative to thread-origin
    #x/yResReg are coordinates of result-variable
    calc = "\t\tresult{0} += register{1}".format(xResReg + yResReg*xCoarsening, x + stencilRange + (y + stencilRange)*xwidth)
    if(coefftype == "Variable"):
        maskAddress = calcMaskAddress(x, y, xResReg, yResReg)
        calc += "*d_mask[{0}]".format(maskAddress)
    calc += ";\n"
    return calc

#prints the additions for a value loaded from (xGrid, yGrid) for all concerning result-variables
def printAdditions(xGrid, yGrid):
    #x/yGrid is relative grid coordinate as seen from thread ranging from (-stencilRange) to (stencilRange + Elems)
    #x/yelem are coordinates of result-variables ranging from (0,0) to (xCoarsening-1, yCoarsening-1)
    content = ""
    for yelem in range(0, yCoarsening, 1):
        for xelem in range(0, xCoarsening, 1):

            if(neighbourhood == "Box"):
                #if chebyshev distance/maximum metric to elem <= stencilRange: add to register
                if((abs(xGrid - xelem) <= stencilRange) and (abs(yGrid - yelem) <= stencilRange)):
                    #if grid-point is first that is added to this elem, then don't add but initialize
                    if(xelem - stencilRange == xGrid and yelem - stencilRange == yGrid):
                        content += printInit(xGrid, yGrid, xelem, yelem)
                    else:
                        content += printSingleAddition(xGrid, yGrid, xelem, yelem)

            if(neighbourhood == "Star"):
                #if point is on same x or y-row and distance is right: add to register
                if((abs(xGrid - xelem) <= stencilRange and yelem == yGrid) or ((abs(yGrid - yelem) <= stencilRange) and xelem == xGrid)):
                    #if grid-point is 'first' that is added to this elem don't add but initialize
                    if(orientation == "Column"):
                        if(yGrid == yelem and xelem - stencilRange == xGrid):
                            content += printInit(xGrid, yGrid, xelem, yelem)
                        else:
                            content += printSingleAddition(xGrid, yGrid, xelem, yelem)
                    if(orientation == "Row"):
                        if(xGrid == xelem and yelem - stencilRange == yGrid):
                            content += printInit(xGrid, yGrid, xelem, yelem)
                        else:
                            content += printSingleAddition(xGrid, yGrid, xelem, yelem)
    return content

def processElement(x, y):
    code = ""
    code += printRegLoad(x, y)
    code += printAdditions(x, y)
    return code

def writeBack(x, y):
    wb = "\t\tarray1[x + {0} + (y + {1})*xdim] = result{2}".format(x, y, x + y*xCoarsening)
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
content += "//Stencil: {0}, range: {1}, xCoarsening: {2}, yCoarsening: {3}, orientation: {4}, alignment: {5}, cacheline size: {6}\n".format(neighbourhood, stencilRange, xCoarsening, yCoarsening, orientation, alignment, CLsize)
content += "#define range {0}\n".format(stencilRange)

if(coefftype == "Variable"):
    content += "\n__constant__ double d_mask[{0}];\n\n".format(stencilSize)

content += "#include <hip/hip_runtime.h>\n"
content += printFunctionHeader()
content += printFunctionInit()
content += printCheck()
    
#for each value in grid:
#   1. load value in variable('register')
#   2. add variable to all result-registers it needs to be added to
#write back results

content += "\t\t//grid is traversed according to access orientation:\n"
if(orientation == "Column"):
    content += "\t\t//column by column\n"
if(orientation == "Row"):
    content += "\t\t//row by row\n"
content += "\n"

if(neighbourhood == "Box"):
    if(orientation == "Column"):
        for x in range(-1*stencilRange, stencilRange + xCoarsening, 1):
            content += "\t\t//column x = {0}\n".format(x)
            for y in range(-1*stencilRange, stencilRange + yCoarsening, 1):
                content += processElement(x, y)
            content += "\n"
    if(orientation == "Row"):
        for y in range(-1*stencilRange, stencilRange + yCoarsening, 1):
            content += "\t\t//row y = {0}\n".format(y)
            for x in range(-1*stencilRange, stencilRange + xCoarsening, 1):
                content += processElement(x, y)
            content += "\n"

elif(neighbourhood == "Star"):
    if(orientation == "Column"):
        #horizontal line left part without intersection
        for x in range(-1*stencilRange, 0, 1):
            content += "\t\t//column x = {0}\n".format(x)
            for y in range(0, yCoarsening, 1):
                content += processElement(x, y)
            content += "\n"
        #vertical line in middle of stencil
        for x in range(0, xCoarsening, 1):
            content += "\t\t//column x = {0}\n".format(x)
            for y in range(-1*stencilRange, stencilRange + yCoarsening, 1):
                content += processElement(x, y)
            content += "\n"
        #horizontal line right part without intersection
        for x in range(xCoarsening, stencilRange + xCoarsening, 1):
            content += "\t\t//column x = {0}\n".format(x)
            for y in range(0, yCoarsening, 1):
                content += processElement(x, y)
            content += "\n"

    if(orientation == "Row"):
        #upper part
        for y in range(-1*stencilRange, 0, 1):
            content += "\t\t//row y = {0}\n".format(y)
            for x in range(0, xCoarsening, 1):
                content += processElement(x, y)
            content += "\n"
        #horizontal line in the middle
        for y in range(0, yCoarsening, 1):
            content += "\t\t//row y = {0}\n".format(y)
            for x in range(-1*stencilRange, stencilRange + xCoarsening, 1):
                content += processElement(x, y)
            content += "\n"
        #lower part
        for y in range(yCoarsening, stencilRange + yCoarsening, 1):
            content += "\t\t//row y = {0}\n".format(y)
            for x in range(0, xCoarsening, 1):
                content += processElement(x, y)
            content += "\n"

content += "\n\t\t//write back\n"
if(coefftype == "Jacobi"):
    if(neighbourhood == "Box"):
        content += "\t\tconst double c = {0}; //1.0/((2*stencilRange + 1)**2)\n".format(1.0/((2*stencilRange + 1)**2))
        content += "\t\t//const double c = 1.0/{0}.0;\n".format((2*stencilRange + 1)**2)
    elif(neighbourhood == "Star"):
        content += "\t\tconst double c = {0}; //1.0/(4*stencilRange + 1)\n".format(1.0/(4*stencilRange + 1))
        content += "\t\t//const double c = 1.0/{0}.0;\n".format(4*stencilRange + 1)

if(orientation == "Column"):
    for x in range(0, xCoarsening, 1):
        for y in range(0, yCoarsening, 1):
            content += writeBack(x, y)
if(orientation == "Row"):
    for y in range(0, yCoarsening, 1):
        for x in range(0, xCoarsening, 1):
            content += writeBack(x, y)

content += "\t}\n"
content += "}\n"
content += "\n"

#init-function
content += "__global__ void init(double * array, const unsigned long xdim, const unsigned long ydim"
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
content += "\tif(x < xdim "
if(alignment == "Read" or alignment == "Write"):
    content += "- rightOffset "
content += "&& y < ydim){\n"
content += "\t\tconst size_t idx = x + y*xdim;\n"
content += "\t\tif(x "
if(alignment == "Write"):
    content += "- {0} ".format(leftOffset)
content += "< range){\n"
content += "\t\t\tarray[idx] = 1.0;\n"
content += "\t\t} else if(y < range){\n"
content += "\t\t\tarray[idx] = 1.0;\n"
content += "\t\t} else{\n"
content += "\t\t\tarray[idx] = 0.0;\n"
content += "\t\t}\n"
content += "\t}\n"
content += "}"
print(content)
