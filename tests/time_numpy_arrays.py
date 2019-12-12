from time import time

import numpy

SIZE = 10000
SUBSIZE = 224 * 224 * 3

start = time()
arr = numpy.zeros((SIZE, SUBSIZE))
for i in range(SIZE):
    arr[i, :] = numpy.ndarray((SUBSIZE, ))
print(f"Replacement: {round(time() - start, 3)}")

start = time()
arr = []
for i in range(SIZE):
    arr.append(numpy.ndarray((SUBSIZE, )))
arr = numpy.hstack(arr)
print(f"Stack: {round(time() - start, 3)}")

