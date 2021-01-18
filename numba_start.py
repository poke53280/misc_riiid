

from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

print(go_fast(x))


from numba import cuda, float32

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp



numba.__version__


from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# Host code
data = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)

integer_array.dtype = np.float32




a = np.empty((1000000, 5000), dtype = np.float32)

def sums_threshold(a):
    res = np.empty(a.shape[0], dtype = np.float32)
    for i in range(a.shape[0]):
        m = np.cumsum(a[i, :]) > 3
        res[i] = np.mean(a[i, :][~m])
    return res


@jit(nopython=True)
def sums_threshold_numba(a):
    res = np.empty(a.shape[0], dtype = np.float32)
    for i in range(a.shape[0]):
        m = np.cumsum(a[i, :]) > 3
        res[i] = np.mean(a[i, :][~m])
    return res


from time import perf_counter

t_begin = perf_counter()

n = sums_threshold(a)

t_pam = perf_counter()

nn = sums_threshold_numba(a)

t_pim = perf_counter()

print("Python", t_pam- t_begin)
print("Numba", t_pim - t_pam)


a.dtype = np.float32
a.dtype = np.int32



diff_timestamp_t = 500 + 1
ptime_t = 150 + 1
user_answer_t = 3 + 1
content_id_t = 13522 + 1

assert np.log2(ptime_t) + np.log2(diff_timestamp_t) + np.log2(user_answer_t) + np.log2(content_id_t) < 32.0

# I incrementally stored approximate delta time stamp, processing time, actual user answer and question id packed into a single uint32 per user event,
# (4 bytes) using the following technique in this competion.
#
# This data packing algorithm is both fast and it packs densely - denser than a fixed this-many-bits-per-feature scheme. It may come as a surprise to some that
# it is even possible to match naive bit-sized spacing with a simple, more readable scheme.
# This is a known technique, and not in any way my own invention.
#
# Do note that all code below works vectorized, typically processing full df_iter's in single commands.
#
# Stage 1: Data scientifically set the comproimse: How much space should each feature deserve.
#
#    Here:
#        I clamp timestamp difference to 0 to 500,000 ms with second resolution. => Values [0, 500] (after division, too).
#        Processing time. I am happy with a range of 0 to 150,000 ms with second resolution. => Values [0, 150]
#        User answer. I store the actual user answer [0..3] in full. (More common was likely to store only true/false.) Values => [0, 3]
#        Question => [0, 13522]

# This determines the following fixed values:
#
# diff_timestamp_t = 500 + 1
# ptime_t = 150 + 1
# user_answer_t = 3 + 1
# content_id_t = 13522 + 1

assert np.log2(ptime_t) + np.log2(diff_timestamp_t) + np.log2(user_answer_t) + np.log2(content_id_t) < 32.0





# Stage 1:
#









