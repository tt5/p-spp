import sys
import os
import random
import math
from numba import njit, prange
import networkx as nx
import imageio
import numpy as np
from scipy.spatial import cKDTree

@njit
def tumble_njit(spile):
    if (spile > 3).any():
        tumbled, spile = np.divmod(spile, 4)
        spile[:-1, :] += tumbled[1:, :]
        spile[1:, :] += tumbled[:-1, :]
        spile[:, :-1] += tumbled[:, 1:]
        spile[:, 1:] += tumbled[:, :-1]
    else:
        #lut = _random_permute4()
        lut = arr = np.array([3, 2, 1, 0])
        result = lut[spile] + 1
        spile[:] = result
        tumbled, spile = np.divmod(spile, 4)
        spile[:-1, :] += tumbled[1:, :]
        spile[1:, :] += tumbled[:-1, :]
        spile[:, :-1] += tumbled[:, 1:]
        spile[:, 1:] += tumbled[:, :-1]
    return spile

@njit
def _random_permute4():
    """Numba-compatible random permutation of [3,2,1,0]."""
    arr = np.array([3, 2, 1, 0])
    for i in range(3, 0, -1):
        j = np.random.randint(0, i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr

@njit(parallel=True)
def tumble_tiles_parallel_njit(T, size, ss, off_y, off_x):
    tiles_y = size // ss
    tiles_x = size // ss
    total = tiles_y * tiles_x
    for i in prange(total):
        y0 = (i // tiles_x) * ss + off_y
        x0 = (i % tiles_x) * ss + off_x
        y1 = min(y0 + ss, size)
        x1 = min(x0 + ss, size)
        if y1 > y0 and x1 > x0 and y0 >= 0 and x0 >= 0 and y0 < size and x0 < size:
            sub = T[y0:y1, x0:x1].copy()
            T[y0:y1, x0:x1] = tumble_njit(sub)


# Pass 2a — vertical seams, parallel over seam index
# Skip crosspoints with horizontal seams (rows yT/yB) plus the start/end
# points (y=0, y=size-1) so the two passes don't race on intersection cells
# and border cells are left for the wrap passes.
@njit(parallel=True)
def boundary_vertical_njit(T, size, ss):
    for j in prange(1, size // ss):
        xL = j*ss - 1
        xR = j*ss
        for y in range(size):
            # skip border rows (y=0, y=size-1) and horizontal-seam rows
            # y%ss==0 catches y=0,ss,2*ss,...
            # y%ss==ss-1 catches y=ss-1,2*ss-1,...,size-1
            if y % ss == 0 or y % ss == ss - 1:
                continue
            T[y, xL] = (T[y, xL] + T[y, xR]) % 4
            T[y, xR] = T[y, xL]

# Pass 2b — horizontal seams, parallel over seam index
# Skip crosspoints with vertical seams (cols xL/xR) plus the start/end
# points (x=0, x=size-1) so the two passes don't race on intersection cells
# and border cells are left for the wrap passes.
@njit(parallel=True)
def boundary_horizontal_njit(T, size, ss):
    for i in prange(1, size // ss):
        yT = i*ss - 1
        yB = i*ss
        for x in range(size):
            # skip border cols (x=0, x=size-1) and vertical-seam cols
            # x%ss==0 catches x=0,ss,2*ss,...
            # x%ss==ss-1 catches x=ss-1,2*ss-1,...,size-1
            if x % ss == 0 or x % ss == ss - 1:
                continue
            T[yT, x] = (T[yT, x] + T[yB, x]) % 4
            T[yB, x] = T[yT, x]


@njit(parallel=True)
def wrap_borderv_njit(T, size):
    T[:, 0] = (T[:, 0] + T[:, size - 1]) % 4
    T[:, size - 1] = T[:, 0]

@njit(parallel=True)
def wrap_borderh_njit(T, size):
    T[0, :] = (T[0, :] + T[size - 1, :]) % 4
    T[size - 1, :] = T[0, :]


#VIDEO_W, VIDEO_H = 192, 108
VIDEO_FPS = 30

video_out_path = "new_video.mp4"
writer = imageio.get_writer(
    video_out_path,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=10,
    pixelformat="yuv444p",
    macro_block_size=None,
)

ss = 16
size = ss*3

VIDEO_W, VIDEO_H = size, size

C = np.zeros((size,size), dtype='int')
C = C+2
bsize = ss//2
if size%2==0:
    C[size//2-bsize:size//2+bsize, size//2-bsize:size//2+bsize] = 4
    #C[size//2-bsize:size//2+bsize, size//2-bsize+ss:size//2+bsize+ss] = 0
    #C[size//2-bsize:size//2+bsize, size//2-bsize-ss:size//2+bsize-ss] = 0
else:
    C[size//2-bsize:size//2+bsize+1, size//2-bsize:size//2+bsize+1] = 0
#
#bsize = 1
#if size%2==0:
#    C[size//2-bsize:size//2+bsize, size//2-bsize:size//2+bsize] = 0
#    #C[size//2-bsize:size//2+bsize, size//2-bsize+ss:size//2+bsize+ss] = 0
#    #C[size//2-bsize:size//2+bsize, size//2-bsize-ss:size//2+bsize-ss] = 0
#else:
#    C[size//2-bsize:size//2+bsize+1, size//2-bsize:size//2+bsize+1] = 8

#C[:ss, :] = 1
#C[:, :ss] = 1
#C[-ss:, :] = 1
#C[:, -ss:] = 1

#C[1:2, 1:-1] = 0
#C[1:-1, 1:2] = 0
#C[-2:-1, 1:-1] = 0
#C[1:-1, -2:-1] = 0

OFFY = (VIDEO_H - size) // 3
OFFX = (VIDEO_W - size) // 3

out   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
frame = np.zeros((size, size, 3), dtype=np.uint8)

#_COL_0 = np.array([0, 0, 0], dtype=np.uint8)
#_COL_1 = np.array([0, 255, 0], dtype=np.uint8)
#_COL_2 = np.array([255, 255, 255], dtype=np.uint8)
#_COL_3 = np.array([255, 0, 0], dtype=np.uint8)
#_COL_4 = np.array([0, 0, 255], dtype=np.uint8)
_COL_0 = np.array([0, 0, 0], dtype=np.uint8)
_COL_1 = np.array([28, 28, 28], dtype=np.uint8)
_COL_2 = np.array([42, 42, 42], dtype=np.uint8)
_COL_3 = np.array([59, 59, 59], dtype=np.uint8)
_COL_4 = np.array([255, 255, 0], dtype=np.uint8)
_COLORS = np.array([_COL_0, _COL_1, _COL_2, _COL_3, _COL_4], dtype=np.uint8)

framecount = 0
print("time,", "0,", "1,", "2,", "3,")
for tick in range(6000):
    #if tick == 400:
    #    C[size//2-1:size//2+1, size//2-1:size//2+1] = 6

    tumble_tiles_parallel_njit(C, size, ss, 0, 0)
    boundary_vertical_njit(C, size, ss)              # Pass 2a
    wrap_borderv_njit(C, size)                        # outer wrap
    boundary_horizontal_njit(C, size, ss)            # Pass 2b
    wrap_borderh_njit(C, size)                        # outer wrap

    if tick%1==0 and tick>=0:
        framecount += 1

        np.take(_COLORS, np.clip(C, 0, 4), axis=0, out=frame)

        out[OFFY:OFFY+size, OFFX:OFFX+size, :] = frame

        n0 = np.sum(C == 0)
        n1 = np.sum(C == 1)
        n2 = np.sum(C == 2)
        n3 = np.sum(C == 3)
        print(framecount, ",", n0, ",", n1, ",",  n2, ",",  n3)

        writer.append_data(out)
        writer.append_data(out)

writer.close()
