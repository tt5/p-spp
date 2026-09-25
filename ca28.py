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
        lut = np.random.permutation(np.array([3, 2, 1, 0]))   # 0→3, 1→2, 2→1, 3→0
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
ss = 4

C = np.zeros((size,size), dtype='int8')

OFFY = (VIDEO_H - size) // 2
OFFX = (VIDEO_W - size) // 2

out   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
frame = np.zeros((size, size, 3), dtype=np.uint8)

_COL_0 = np.array([0, 0, 0], dtype=np.uint8)
_COL_1 = np.array([0, 255, 0], dtype=np.uint8)
_COL_2 = np.array([255, 255, 255], dtype=np.uint8)
_COL_3 = np.array([255, 0, 0], dtype=np.uint8)
_COLORS = np.array([_COL_0, _COL_1, _COL_2, _COL_3], dtype=np.uint8)

framecount = 0
print("time,", "0,", "1,", "2,", "3,")
for tick in range(300):

    tumble_tiles_parallel_njit(C, size, ss, 0, 0)

    if tick%1==0 and tick>0:
        framecount += 1

        np.take(_COLORS, C, axis=0, out=frame)

        out[OFFY:OFFY+size, OFFX:OFFX+size, :] = frame

        n0 = np.sum(C == 1)
        n1 = np.sum(C == 1)
        n2 = np.sum(C == 2)
        n3 = np.sum(C == 3)
        print(framecount, ",", n0, ",", n1, ",",  n2, ",",  n3)

        writer.append_data(out)

writer.close()
np.save('frak2', C)
