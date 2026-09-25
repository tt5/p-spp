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
@njit(parallel=True)
def boundary_vertical_njit(T, size, ss):
    for j in prange(1, size // ss):
        xL = j*ss - 1
        xR = j*ss
        T[:, xL] = (T[:, xL] + T[:, xR])%4
        T[:, xR] = T[:, xL]

# Pass 2b — horizontal seams, parallel over seam index
@njit(parallel=True)
def boundary_horizontal_njit(T, size, ss):
    for i in prange(1, size // ss):
        yT = i*ss - 1
        yB = i*ss
        T[yT, :] = (T[yT, :] + T[yB, :])%4
        T[yB, :] = T[yT, :]


@njit(parallel=True)
def wrap_border_njit(T, size):
    # outer wrap: left col touches right col, top row touches bottom row
    # same rule as internal seams: both sides become (a+b)%4
    T[:, 0] = (T[:, 0] + T[:, size - 1]) % 4
    T[:, size - 1] = T[:, 0]
    T[0, :] = (T[0, :] + T[size - 1, :]) % 4
    T[size - 1, :] = T[0, :]


#VIDEO_W, VIDEO_H = 1920, 1080
VIDEO_W, VIDEO_H = 960, 540
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

size = 6*64
ss = 64

C = np.zeros((size,size), dtype='int')
C = C+4
#bsize = size//8
#C[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 0
C[:1, :] = 0
C[:, :1] = 0
C[-1:, :] = 0
C[:, -1:] = 0

OFFY = (VIDEO_H - size) // 2
OFFX = (VIDEO_W - size) // 2

out   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
frame = np.zeros((size, size, 3), dtype=np.uint8)

#_COL_0 = np.array([0, 0, 0], dtype=np.uint8)
#_COL_1 = np.array([0, 255, 0], dtype=np.uint8)
#_COL_2 = np.array([255, 255, 255], dtype=np.uint8)
#_COL_3 = np.array([255, 0, 0], dtype=np.uint8)
#_COL_4 = np.array([0, 0, 255], dtype=np.uint8)
_COL_0 = np.array([0, 0, 0], dtype=np.uint8)
_COL_1 = np.array([16, 16, 16], dtype=np.uint8)
_COL_2 = np.array([24, 24, 24], dtype=np.uint8)
_COL_3 = np.array([31, 31, 31], dtype=np.uint8)
_COL_4 = np.array([255, 255, 0], dtype=np.uint8)
_COLORS = np.array([_COL_0, _COL_1, _COL_2, _COL_3, _COL_4], dtype=np.uint8)

framecount = 0
print("time,", "0,", "1,", "2,", "3,")
for tick in range(1500):

    tumble_tiles_parallel_njit(C, size, ss, 0, 0)
    boundary_vertical_njit(C, size, ss)              # Pass 2a
    boundary_horizontal_njit(C, size, ss)            # Pass 2b
    wrap_border_njit(C, size)                        # outer wrap

    if tick%1==0 and tick>=1300:
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
        writer.append_data(out)

writer.close()
np.save('frak2', C)
