import numpy as np
import math
import sys
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import imageio
import time
from numba import njit, prange

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

writer = imageio.get_writer('new_video.mp4', fps=60, macro_block_size=1)
def do_add( spile, tumbled ):
    """ Updates spile in place """
    spile[ :-1, :] += tumbled[ 1:, :] # Shift N and add                 
    spile[ 1:, :] += tumbled[ :-1, :] # Shift S   
    spile[ :, :-1] += tumbled[ :, 1:] # Shift W
    spile[ :, 1:] += tumbled[ :, :-1] # Shift E

def tumble( spile ):
    while ( spile > 3 ).any():
        tumbled, spile = np.divmod( spile, 4 )
        do_add( spile, tumbled )
    return spile

@njit
def do_add_njit(spile, tumbled):
    spile[:-1, :] += tumbled[1:, :]
    spile[1:, :] += tumbled[:-1, :]
    spile[:, :-1] += tumbled[:, 1:]
    spile[:, 1:] += tumbled[:, :-1]

@njit
def tumble_njit(spile):
    while (spile > 3).any():
        tumbled, spile = np.divmod(spile, 4)
        do_add_njit(spile, tumbled)
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
def promote_queens_njit(C, AC, ND, size):
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
def remove_queens_njit(C, ND, size):
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
def add_queen_energy_njit(C, ND, AC, size, qenergy):
    ys, xs = np.where(C == 3)
    for k in range(ys.shape[0]):
        y = ys[k]
        x = xs[k]
        ND[y, x] = ND[y, x] + qenergy

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
                if killed == True:
                    break
                if pkill < 0.5:
                    isPkill = False
                    break
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4
                killed = True
                break
        for i in range(8):
            j = (north, south, east, west, northeast, northwest, southeast, southwest)[i]
            if j == 5 and nC>=3:
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4
                break

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

@njit
def birthAC_njit(C, AC, size):
    ys, xs = np.where(AC > 0)
    for k in range(ys.shape[0]):
        y = ys[k]
        x = xs[k]
        if C[y,x] == 2 or C[y,x] == 3 or C[y,x] == 1:
            continue
        north = south = east = west = 0
        if y>0: north = C[y-1,x]
        if y<size-1: south = C[y+1,x]
        if x<size-1: east = C[y,x+1]
        if x>0: west = C[y,x-1]
        nA = 0
        nC = 0
        nN = 0
        if north == 4: nA += 1
        elif north == 5: nC += 1
        elif north == 2: nN += 1
        if south == 4: nA += 1
        elif south == 5: nC += 1
        elif south == 2: nN += 1
        if east == 4: nA += 1
        elif east == 5: nC += 1
        elif east == 2: nN += 1
        if west == 4: nA += 1
        elif west == 5: nC += 1
        elif west == 2: nN += 1
        if nA > nC:
            C[y,x] = 4
        elif nA < nC:
            C[y,x] = 5
        else:
            if nN > 0:
                C[y,x] = 5
            else:
                C[y,x] = 4


def compute_param_grids(nodes, size, k=2):
    """Compute per-cell (a0, alpha, gamma, cAA, cAC, cCC) by averaging k nearest nodes."""
    if len(nodes) == 0:
        return np.zeros((size, size, 6), dtype='float64')
    nodes_arr = np.array(nodes, dtype='float64')
    node_xy = nodes_arr[:, :2]
    node_params = nodes_arr[:, 2:]
    cell_coords = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing='ij'), axis=-1).reshape(-1, 2).astype('float64')
    diffs = cell_coords[:, None, :] - node_xy[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    knn = min(k, len(nodes))
    knn_idx = np.argpartition(dists, knn, axis=1)[:, :knn]
    avg_params = node_params[knn_idx].mean(axis=1)
    return avg_params.reshape(size, size, 6)


size = 256

C = np.zeros((size,size), dtype='int')
ND = np.zeros((size,size), dtype='int')
AC = np.zeros((size,size), dtype='int')

# 1 queen
# 2 N
# 3 D
# 4 A
# 5 C

# Parameter graph nodes: [x, y, a0, alpha, gamma, cAA, cAC, cCC]
# Default: 4 corners
nodes = [
    [0, 0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    [size-1, 0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0, size-1, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
    [size-1, size-1, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
]

# Initial conditions
bsize = 4
C[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C[0, 0:size-1] = 4
C[0:size-1, 0] = 4
C[size-1, 0:size-1] = 4
C[0:size-1, size-1] = 4

k_nearest = 2

# Compute per-cell parameter grids from current node positions
param_grid = compute_param_grids(nodes, size, k=k_nearest)
a0_grid = param_grid[:, :, 0].copy()
alpha_grid = param_grid[:, :, 1].copy()
gamma_grid = param_grid[:, :, 2].copy()
cAA_grid = param_grid[:, :, 3].copy()
cAC_grid = param_grid[:, :, 4].copy()
cCC_grid = param_grid[:, :, 5].copy()

# Grid can be updated mid-simulation by modifying the nodes list and
# re-computing param_grid (e.g., nodes.append([...]) then recompute).

framecount = 0
print("time,", "N,", "D,", "A,", "C")
for tick in range(500):

    promote_queens_njit(C, AC, ND, size)
    remove_queens_njit(C, ND, size)

    birthAC_njit(C, AC, size)
    birthND_njit(C, ND, size)
    eatC_njit(C, size, a0_grid, alpha_grid, gamma_grid, cAC_grid, cCC_grid)
    eatA_njit(C, size, a0_grid, alpha_grid, cAA_grid, cAC_grid)

    if tick == 100:
        nodes.append([size//2, size//2, 0.7, 1.2, 0.8, 1.5, 1.0, 1.0])
        param_grid = compute_param_grids(nodes, size, k=k_nearest)
        a0_grid = param_grid[:, :, 0].copy()
        alpha_grid = param_grid[:, :, 1].copy()
        gamma_grid = param_grid[:, :, 2].copy()
        cAA_grid = param_grid[:, :, 3].copy()
        cAC_grid = param_grid[:, :, 4].copy()
        cCC_grid = param_grid[:, :, 5].copy()

    qenergy = 4
    add_queen_energy_njit(C, ND, AC, size, qenergy)

    maxclip = 3 + qenergy
    
    ND = np.clip(ND - ((C != 2) & (C != 3)) * maxclip, 0, maxclip)
    if tick%1==0:
        AC = AC + 1 - ((C != 4) & (C != 5)) * 1

    nd_pre = ND.copy()
    ac_pre = AC.copy()

    ss = random.choice([64])

    if tick%2==0:
        off_y = 0
        off_x = 0
    else:
        off_y = ss//2
        off_x = ss//2

    tumble_tiles_parallel_njit(ND, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC, size, ss, off_y, off_x)

    if tick>0:
        framecount += 1

        out = np.zeros((1080, 1920, 3), dtype='int')
        
        panel_w = size
        total_w = panel_w * 1
        gap = max((1920 - total_w) // 4, 0)
        y0 = (1080 - size) // 2
        x0 = gap
        x1 = gap + panel_w
        x2 = gap + panel_w * 2
        
        # ND panel — species-colored rendering
        ny, nx = y0, x0

        # subtle energy-field background from ND
        e = np.clip(ND * 10, 0, 255).astype('int')
        frame = np.zeros((size, size, 3), dtype='int')
        frame[:, :, 0] = (e // 4).astype('int')
        frame[:, :, 1] = (e // 2).astype('int')
        frame[:, :, 2] = e.astype('int')

        # species palette (R, G, B)
        COL_Q = np.array([255, 224, 110], dtype='int')   # gold
        COL_N = np.array([80, 210, 255], dtype='int')    # cyan
        COL_D = np.array([55, 115, 255], dtype='int')    # blue
        COL_A = np.array([255, 150, 70], dtype='int')    # orange
        COL_C = np.array([230, 70, 180], dtype='int')    # magenta

        # overlay species colors (bright, mixed over background)
        for mask, col in [(C == 1, COL_Q), (C == 2, COL_N),
                          (C == 3, COL_D), (C == 4, COL_A),
                          (C == 5, COL_C)]:
            frame[mask] = np.clip(frame[mask] * 0.3 + col * 0.7, 0, 255)

        # faint motion trail from previous ND frame (soft)
        trail = np.clip(nd_pre * 6, 0, 255).astype('int')
        frame[:, :, 0] = (frame[:, :, 0] * 0.9 + trail * 0.1).astype('int')
        frame[:, :, 1] = (frame[:, :, 1] * 0.9 + trail * 0.1).astype('int')
        frame[:, :, 2] = (frame[:, :, 2] * 0.9 + trail * 0.1).astype('int')

        frame = np.clip(frame, 0, 255)
        out[ny:ny+size, nx:nx+size] = frame
        
        frame = out

        nN = np.sum(C == 2)
        nD = np.sum(C == 3)
        nA = np.sum(C == 4)
        nC = np.sum(C == 5)
        print(framecount, ",", nN, ",", nD, ",",  nA, ",",  nC)
        
        writer.append_data(np.array(frame, dtype=np.uint8))

writer.close()
