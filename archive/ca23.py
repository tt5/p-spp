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
    for i in range(99999):
        if (spile > 3).any():
            tumbled, spile = np.divmod(spile, 4)
            spile[:-1, :] += tumbled[1:, :]
            spile[1:, :] += tumbled[:-1, :]
            spile[:, :-1] += tumbled[:, 1:]
            spile[:, 1:] += tumbled[:, :-1]
        else:
            #if i == 0:
            #    spile[:] = 2
            break
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
        nA = 0
        nC = 0
        if north == 4:
            nA += 1
        if south == 4:
            nA += 1
        if east == 4:
            nA += 1
        if west == 4:
            nA += 1
        if north == 5:
            nC += 1
        if south == 5:
            nC += 1
        if east == 5:
            nC += 1
        if west == 5:
            nC += 1
        if nA >= 4 or nC >= 4:
            C[y, x] = 5

@njit
def add_energy_njit(C, ND, size, energy):
    ys, xs = np.where(C == 3)
    for k in range(ys.shape[0]):
        y = ys[k]
        x = xs[k]
        ND[y, x] = ND[y, x] + energy

@njit
def _count_neighbors8(north, south, east, west, northeast, northwest, southeast, southwest):
    nD = nA = nC = 0
    if north == 3: nD += 1
    elif north == 4: nA += 1
    elif north == 5: nC += 1
    if south == 3: nD += 1
    elif south == 4: nA += 1
    elif south == 5: nC += 1
    if east == 3: nD += 1
    elif east == 4: nA += 1
    elif east == 5: nC += 1
    if west == 3: nD += 1
    elif west == 4: nA += 1
    elif west == 5: nC += 1
    if northeast == 3: nD += 1
    elif northeast == 4: nA += 1
    elif northeast == 5: nC += 1
    if northwest == 3: nD += 1
    elif northwest == 4: nA += 1
    elif northwest == 5: nC += 1
    if southeast == 3: nD += 1
    elif southeast == 4: nA += 1
    elif southeast == 5: nC += 1
    if southwest == 3: nD += 1
    elif southwest == 4: nA += 1
    elif southwest == 5: nC += 1
    return nD, nA, nC

@njit
def eatA_njit(C, size, a0_grid, alpha_grid, cAA_grid, cAC_grid):
    """Per-cell eatA: pkill computed from spatial parameter grids."""
    ys, xs = np.where(C == 4)
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
        nD, nA, nC = _count_neighbors8(north, south, east, west, northeast, northwest, southeast, southwest)
        denomA = 1 + cAA_grid[y,x]*nA + cAC_grid[y,x]*nC
        pkill = max(0, min((a0_grid[y,x] + alpha_grid[y,x]*nD) / denomA, 1))
        killed = False
        isPkill = True
        for i in range(8):
            j = (north, south, east, west, northeast, northwest, southeast, southwest)[i]
            if j == 2:
                if pkill < 0.5:
                    isPkill = False
                    break
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4
                killed = True
        if isPkill == False or killed == True:
            continue
        for i in range(8):
            j = (north, south, east, west, northeast, northwest, southeast, southwest)[i]
            if j == 3:
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4
                killed = True
        if killed == True:
            continue
        for i in range(8):
            j = (north, south, east, west, northeast, northwest, southeast, southwest)[i]
            if j == 5:
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4

@njit
def eatC_njit(C, size, a0_grid, alpha_grid, gamma_grid, cAC_grid, cCC_grid):
    """Per-cell eatC: pkill computed from spatial parameter grids."""
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
        nD, nA, nC = _count_neighbors8(north, south, east, west, northeast, northwest, southeast, southwest)
        for i in range(8):
            j = (north, south, east, west, northeast, northwest, southeast, southwest)[i]
            if j == 2:
                denomC = 1 + cCC_grid[y,x] - gamma_grid[y,x]*nC + cAC_grid[y,x]*nA
                denomC = max(denomC, 1e-6)
                pkill = max(0, min((a0_grid[y,x] + alpha_grid[y,x]*nD) / denomC, 1))
                if pkill < 0.5:
                    break
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
        if nN >= 3:
            if C[y,x] != 1:
                C[y,x] = 3
        else:
            if C[y,x] != 1:
                C[y,x] = 2


def compute_param_grids(nodes, size, out_a0, out_alpha, out_gamma, out_cAA, out_cAC, out_cCC):
    if len(nodes) == 0:
        out_a0.fill(0); out_alpha.fill(0); out_gamma.fill(0)
        out_cAA.fill(0); out_cAC.fill(0); out_cCC.fill(0)
        return
    node_ids = list(nodes.nodes())
    N_nodes = len(node_ids)
    node_xy = np.empty((N_nodes, 2), dtype='float64')
    node_params = np.empty((N_nodes, 6), dtype='float64')
    for i, nid in enumerate(node_ids):
        nd = nodes.nodes[nid]
        node_xy[i, 0] = nd['x']
        node_xy[i, 1] = nd['y']
        node_params[i, 0] = nd['a0']
        node_params[i, 1] = nd['alpha']
        node_params[i, 2] = nd['gamma']
        node_params[i, 3] = nd['cAA']
        node_params[i, 4] = nd['cAC']
        node_params[i, 5] = nd['cCC']

    # Nearest-neighbour (Voronoi): cKDTree query.
    tree = cKDTree(node_xy)
    _, best_node = tree.query(_coords_array, k=1)

    out_a0.ravel()  [:] = node_params[best_node, 0]
    out_alpha.ravel()[:] = node_params[best_node, 1]
    out_gamma.ravel()[:] = node_params[best_node, 2]
    out_cAA.ravel()  [:] = node_params[best_node, 3]
    out_cAC.ravel()  [:] = node_params[best_node, 4]
    out_cCC.ravel()  [:] = node_params[best_node, 5]
    return

VIDEO_W, VIDEO_H = 288, 162
VIDEO_FPS = 10

video_out_path = "new_video.mp4"
writer = imageio.get_writer(
    video_out_path,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=10,
    pixelformat="yuv444p",
    macro_block_size=None,
)

size = 150

C = np.zeros((size,size), dtype='int8')
ND = np.zeros((size,size), dtype='int')

# Pre-allocated parameter grids — reused across every compute_param_grids call.
a0_grid = np.zeros((size, size), dtype='float64')
alpha_grid = np.zeros((size, size), dtype='float64')
gamma_grid = np.zeros((size, size), dtype='float64')
cAA_grid = np.zeros((size, size), dtype='float64')
cAC_grid = np.zeros((size, size), dtype='float64')
cCC_grid = np.zeros((size, size), dtype='float64')

# Precomputed per-cell coordinate vectors (row-major, shape (size*size,)).
# _cell_x[i] = x-coord of cell i, _cell_y[i] = y-coord of cell i.
# Depends only on `size`, which is constant, so built once.
_cell_x = np.tile(np.arange(size, dtype='float64'), size)
_cell_y = np.repeat(np.arange(size, dtype='float64'), size)
# Precomputed coordinate array for cKDTree query — built once.
_coords_array = np.column_stack([_cell_x, _cell_y])

# 1 queen
# 2 N
# 3 D
# 4 A
# 5 C

nodes = nx.complete_graph(4)

for i, (x, y) in enumerate([(0, 0), (size-1, 0), (0, size-1), (size-1, size-1)]):
    nodes.nodes[i].update({
        'x': x, 'y': y,
        'a0': 0.5, 'alpha': 1.0, 'gamma': 1.0,
        'cAA': 1.0, 'cAC': 1.0, 'cCC': 1.0,
        'chips': 0,
    })

# Initial conditions
C = C+3
C[63:66, 1:-1] = 4
C[60:62, 1:-1] = 5
C[1:-1, 63:66] = 4
C[1:-1, 60:62] = 5
C[-3:-1, 1:-1] = 4
C[1:-1, -3:-1] = 4

# Compute per-cell parameter grids from current node positions
compute_param_grids(nodes, size, a0_grid, alpha_grid, gamma_grid, cAA_grid, cAC_grid, cCC_grid)

a0 = 0.5
alpha = 1.8
gamma = 1.0
cAA = 8.0
cAC = 2.0
cCC = 1.0

INJECTION_SCHEDULE = {
    42: (size//2,     size//2),
}

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
print("time,", "N,", "D,", "A,", "C")
for tick in range(250):

    promote_queens_njit(C, size)
    remove_queens_njit(C, size)

    birthND_njit(C, ND, size)
    eatC_njit(C, size, a0_grid, alpha_grid, gamma_grid, cAC_grid, cCC_grid)
    eatA_njit(C, size, a0_grid, alpha_grid, cAA_grid, cAC_grid)

    inject = INJECTION_SCHEDULE.get(tick)
    if inject is not None:
        x, y = inject
        new_id = len(nodes)
        nodes.add_node(new_id, x=x, y=y, a0=a0, alpha=alpha, gamma=gamma,
                       cAA=cAA, cAC=cAC, cCC=cCC)
        for existing in range(new_id):
            nodes.add_edge(new_id, existing)
        nodes.nodes[new_id]['chips'] = 0
        compute_param_grids(nodes, size, a0_grid, alpha_grid,
                            gamma_grid, cAA_grid, cAC_grid, cCC_grid)

    add_energy_njit(C, ND, size, 4)
    ND[~((C == 2) | (C == 3))] = 0
    tumble_tiles_parallel_njit(ND, size, 50, 0, 0)

    if tick%1==0 and tick>0:
        framecount += 1

        np.take(_COLORS, C, axis=0, out=frame)

        out[OFFY:OFFY+size, OFFX:OFFX+size, :] = frame

        print(tick)

        writer.append_data(out)

writer.close()
