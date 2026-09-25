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
def promote_queens_njit(C, size):
    ys, xs = np.where(C == 3)
    for k in range(ys.shape[0]):
        y = ys[k]
        x = xs[k]
        north = south = east = west = 0
        if y > 0:
            north = C[y - 1, x]
        if y < size - 1:
            south = C[y + 1, x]
        if x < size - 1:
            east = C[y, x + 1]
        if x > 0:
            west = C[y, x - 1]
        nN = 0
        if north == 2:
            nN += 1
        if south == 2:
            nN += 1
        if east == 2:
            nN += 1
        if west == 2:
            nN += 1
        if nN >= 4:
            C[y, x] = 1

@njit
def birthND_njit(C, size):
    ys, xs = np.indices((size, size)).reshape(2, -1)

    for k in range(size*size):
        y = ys[k]
        x = xs[k]
        north = south = east = west = 0
        if y>0: north = C[y-1,x]
        if y<size-1: south = C[y+1,x]
        if x<size-1: east = C[y,x+1]
        if x>0: west = C[y,x-1]
        nN = 0
        if north == 2: nN += 1
        if south == 2: nN += 1
        if east == 2: nN += 1
        if west == 2: nN += 1
        if nN == 3:
            C[y,x] = 3
        else:
            if C[y,x] != 1:
                C[y,x] = 2


VIDEO_W, VIDEO_H = 1920, 1080
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

size = 1024
ss = 1024

C = np.zeros((size,size), dtype='int8')

# 1 queen
# 2 N
# 3 D

# Initial conditions
C = C+3

OFFY = (VIDEO_H - size) // 2
OFFX = (VIDEO_W - size) // 2

out   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
frame = np.zeros((size, size, 3), dtype=np.uint8)

_COL_Q = np.array([0, 255, 0], dtype=np.uint8)
_COL_D = np.array([255, 255, 255], dtype=np.uint8)
_COL_N = np.array([255, 0, 0], dtype=np.uint8)
_COL_E = np.array([0, 0, 0], dtype=np.uint8)
_COLORS = np.array([_COL_E, _COL_Q, _COL_N, _COL_D], dtype=np.uint8)

framecount = 0
print("time,", "N,", "D,", "Q,")
for tick in range(300):

    promote_queens_njit(C, size)

    birthND_njit(C, size)

    if tick%1==0 and tick>0:
        framecount += 1

        np.take(_COLORS, C, axis=0, out=frame)

        out[OFFY:OFFY+size, OFFX:OFFX+size, :] = frame

        nN = np.sum(C == 2)
        nD = np.sum(C == 3)
        nQ = np.sum(C == 1)
        print(framecount, ",", nN, ",", nD, ",",  nQ)

        writer.append_data(out)

writer.close()
np.save('frak2', C)
