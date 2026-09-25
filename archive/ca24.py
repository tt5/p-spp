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
        lut = np.array([3, 2, 1, 0])   # 0→3, 1→2, 2→1, 3→0
        result = lut[spile] + 1
        spile[:] = result.astype(np.int8)
    return spile

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
def remove_queens_njit(C, size):
    ys, xs = np.where(C == 1)
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
        nC = 0
        if north == 5:
            nC += 1
        if south == 5:
            nC += 1
        if east == 5:
            nC += 1
        if west == 5:
            nC += 1
        if nC >= 4:
            C[y, x] = 5

@njit
def add_energy_njit(C, ND, size, energy):
    ys, xs = np.where(C == 3)
    for k in range(ys.shape[0]):
        y = ys[k]
        x = xs[k]
        ND[y, x] = 4

@njit
def eatC_njit(C, size):
    ys, xs = np.where(C == 5)
    deltas = ((-1,0),(1,0),(0,1),(0,-1),(-1,1),(-1,-1),(1,1),(1,-1))
    for k in range(ys.shape[0]):
        y = ys[k]
        x = xs[k]
        north = south = east = west = northeast = northwest = southeast = southwest = 0
        if y>0: north = C[y-1,x]
        if y<size-1: south = C[y+1,x]
        if x<size-1: east = C[y,x+1]
        if x>0: west = C[y,x-1]
        if y>0 and x<size-1: northeast = C[y-1,x+1]
        if y>0 and x>0: northwest = C[y-1,x-1]
        if y<size-1 and x<size-1: southeast = C[y+1,x+1]
        if y<size-1 and x>0: southwest = C[y+1,x-1]
        for i in range(8):
            j = (north, south, east, west, northeast, northwest, southeast, southwest)[i]
            if j == 2:
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 5

@njit
def birthND_njit(C, ND, size):
    ys, xs = np.where(ND > 0)
    for k in range(ys.shape[0]):
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


#VIDEO_W, VIDEO_H = 1920, 1080
VIDEO_W, VIDEO_H = 800, 800
VIDEO_FPS = 60

video_out_path = "new_video.mp4"
writer = imageio.get_writer(
    video_out_path,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=10,
    pixelformat="yuv444p",
    macro_block_size=None,
)

size = 800
ss = 8

C = np.zeros((size,size), dtype='int8')
ND = np.zeros((size,size), dtype='int')

# 1 queen
# 2 N
# 3 D
# 4 A
# 5 C

# Initial conditions
C = C+3

C[-5:-2, :] = 5

OFFY = (VIDEO_H - size) // 2
OFFX = (VIDEO_W - size) // 2

out   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
frame = np.zeros((size, size, 3), dtype=np.uint8)

_COL_Q = np.array([0, 255, 0], dtype=np.uint8)
_COL_D = np.array([255, 255, 255], dtype=np.uint8)
_COL_N = np.array([255, 0, 0], dtype=np.uint8)
_COL_A = np.array([0, 0, 0], dtype=np.uint8)
_COL_C = np.array([0, 0, 255], dtype=np.uint8)
_COL_E = np.array([0, 0, 0], dtype=np.uint8)
_COLORS = np.array([_COL_E, _COL_Q, _COL_N, _COL_D, _COL_A, _COL_C], dtype=np.uint8)

framecount = 0
print("time,", "N,", "D,", "Q,", "C")
for tick in range(1000):

    promote_queens_njit(C, size)
    remove_queens_njit(C, size)

    birthND_njit(C, ND, size)
    #eatC_njit(C, size)

    #print("max: ", np.max(ND))
    #ND[(C == 3)] = 5
    #ND[(C == 5)] = 0
    tumble_tiles_parallel_njit(ND, size, ss, 0, 0)

    if tick%1==0 and tick>0:
        framecount += 1

        np.take(_COLORS, C, axis=0, out=frame)

        out[OFFY:OFFY+size, OFFX:OFFX+size, :] = frame

        nN = np.sum(C == 2)
        nD = np.sum(C == 3)
        nQ = np.sum(C == 1)
        nC = np.sum(C == 5)
        print(framecount, ",", nN, ",", nD, ",",  nQ, ",",  nC)

        writer.append_data(out)

writer.close()
np.save('frak2', C)
