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
def eatA_njit(C, ND, AC, size, a0, alpha, cAA, cAC):
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
        neighbors = (north, south, east, west, northeast, northwest, southeast, southwest)
        killed = False
        isPkill = True
        for i in range(8):
            j = neighbors[i]
            if j == 2:
                denomA = 1 + cAA*nA + cAC*nC
                pkill = max(0, min((a0 + alpha*nD) / denomA, 1))
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
            j = neighbors[i]
            if j == 3:
                denomA = 1 + cAA*nA + cAC*nC
                pkill = max(0, min((a0 + alpha*nD) / denomA, 1))
                if pkill < 0.5:
                    isPkill = False
                    break
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4
                if killed == True:
                    break
                killed = True
                break
        for i in range(8):
            j = neighbors[i]
            if j == 5 and nC>=3:
                dy, dx = deltas[i]
                ny = y + dy
                nx = x + dx
                C[ny, nx] = 4
                killed = True
                break

@njit
def eatC_njit(C, ND, size, a0, alpha, gamma, cAC, cCC):
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
        neighbors = (north, south, east, west, northeast, northwest, southeast, southwest)
        for i in range(8):
            j = neighbors[i]
            if j == 2:
                denomC = 1 + cCC - gamma*nC + cAC*nA
                denomC = max(denomC, 1e-6)
                pkill = max(0, min((a0 + alpha*nD) / denomC, 1))
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
        nA = 0
        nC = 0
        nN = 0
        if north == 5: nC += 1
        elif north == 4: nA += 1
        elif north == 2: nN += 1
        if south == 5: nC += 1
        elif south == 4: nA += 1
        elif south == 2: nN += 1
        if east == 5: nC += 1
        elif east == 4: nA += 1
        elif east == 2: nN += 1
        if west == 5: nC += 1
        elif west == 4: nA += 1
        elif west == 2: nN += 1
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


size = 192

C1 = np.zeros((size,size), dtype='int')
ND1 = np.zeros((size,size), dtype='int')
AC1 = np.zeros((size,size), dtype='int')

C2 = np.zeros((size,size), dtype='int')
ND2 = np.zeros((size,size), dtype='int')
AC2 = np.zeros((size,size), dtype='int')

C3 = np.zeros((size,size), dtype='int')
ND3 = np.zeros((size,size), dtype='int')
AC3 = np.zeros((size,size), dtype='int')

C4 = np.zeros((size,size), dtype='int')
ND4 = np.zeros((size,size), dtype='int')
AC4 = np.zeros((size,size), dtype='int')

C5 = np.zeros((size,size), dtype='int')
ND5 = np.zeros((size,size), dtype='int')
AC5 = np.zeros((size,size), dtype='int')

C6 = np.zeros((size,size), dtype='int')
ND6 = np.zeros((size,size), dtype='int')
AC6 = np.zeros((size,size), dtype='int')

C7 = np.zeros((size,size), dtype='int')
ND7 = np.zeros((size,size), dtype='int')
AC7 = np.zeros((size,size), dtype='int')

C8 = np.zeros((size,size), dtype='int')
ND8 = np.zeros((size,size), dtype='int')
AC8 = np.zeros((size,size), dtype='int')

# 1 queen
# 2 N
# 3 D
# 4 A
# 5 C

a0 = 0.5
alpha1 = 1.5 # help from defectors
alpha2 = 1.3 # help from defectors
alpha3 = 1.0 # help from defectors
alpha4 = 0.9 # help from defectors
gamma = 1.0 # cooperation for C

# interference
cAA = 1.0
cAC = 1.0
cCC = 1.0

# initial conditions
bsize = 4
C1[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C2[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C3[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C4[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C5[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C6[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C7[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C8[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3

C1[0, 0] = 4
C1[0, size-1] = 4
C1[size-1, 0] = 4
C1[size-1, size-1] = 4

C2[0, 0] = 4
C2[0, size-1] = 4
C2[size-1, 0] = 4
C2[size-1, size-1] = 4

C3[0, 0] = 4
C3[0, size-1] = 4
C3[size-1, 0] = 4
C3[size-1, size-1] = 4

C4[0, 0] = 4
C4[0, size-1] = 4
C4[size-1, 0] = 4
C4[size-1, size-1] = 4

C5[0, 0] = 4
C5[0, size-1] = 4
C5[size-1, 0] = 4
C5[size-1, size-1] = 4

C6[0, 0] = 4
C6[0, size-1] = 4
C6[size-1, 0] = 4
C6[size-1, size-1] = 4

C7[0, 0] = 4
C7[0, size-1] = 4
C7[size-1, 0] = 4
C7[size-1, size-1] = 4

C8[0, 0] = 4
C8[0, size-1] = 4
C8[size-1, 0] = 4
C8[size-1, size-1] = 4

framecount = 0
print("time,", "N,", "D,", "A,", "C")
for tick in range(1000):

    promote_queens_njit(C1, AC1, ND1, size)
    remove_queens_njit(C1, ND1, size)
    promote_queens_njit(C2, AC2, ND2, size)
    remove_queens_njit(C2, ND2, size)
    promote_queens_njit(C3, AC3, ND3, size)
    remove_queens_njit(C3, ND3, size)
    promote_queens_njit(C4, AC4, ND4, size)
    remove_queens_njit(C4, ND4, size)
    promote_queens_njit(C5, AC5, ND5, size)
    remove_queens_njit(C5, ND5, size)
    promote_queens_njit(C6, AC6, ND6, size)
    remove_queens_njit(C6, ND6, size)
    promote_queens_njit(C7, AC7, ND7, size)
    remove_queens_njit(C7, ND7, size)
    promote_queens_njit(C8, AC8, ND8, size)
    remove_queens_njit(C8, ND8, size)

    birthAC_njit(C1, AC1, size)
    birthND_njit(C1, ND1, size)
    birthAC_njit(C2, AC2, size)
    birthND_njit(C2, ND2, size)
    birthAC_njit(C3, AC3, size)
    birthND_njit(C3, ND3, size)
    birthAC_njit(C4, AC4, size)
    birthND_njit(C4, ND4, size)
    birthAC_njit(C5, AC5, size)
    birthND_njit(C5, ND5, size)
    birthAC_njit(C6, AC6, size)
    birthND_njit(C6, ND6, size)
    birthAC_njit(C7, AC7, size)
    birthND_njit(C7, ND7, size)
    birthAC_njit(C8, AC8, size)
    birthND_njit(C8, ND8, size)
    eatC_njit(C1, ND1, size, a0, alpha1, gamma, cAC, cCC)
    eatA_njit(C1, ND1, AC1, size, a0, alpha1, cAA, cAC)
    eatC_njit(C2, ND2, size, a0, alpha2, gamma, cAC, cCC)
    eatA_njit(C2, ND2, AC2, size, a0, alpha2, cAA, cAC)
    eatC_njit(C3, ND3, size, a0, alpha3, gamma, cAC, cCC)
    eatA_njit(C3, ND3, AC3, size, a0, alpha3, cAA, cAC)
    eatC_njit(C4, ND4, size, a0, alpha4, gamma, cAC, cCC)
    eatA_njit(C4, ND4, AC4, size, a0, alpha4, cAA, cAC)
    eatC_njit(C5, ND5, size, a0, alpha1, gamma, cAC, cCC)
    eatA_njit(C5, ND5, AC5, size, a0, alpha1, cAA, cAC)
    eatC_njit(C6, ND6, size, a0, alpha2, gamma, cAC, cCC)
    eatA_njit(C6, ND6, AC6, size, a0, alpha2, cAA, cAC)
    eatC_njit(C7, ND7, size, a0, alpha3, gamma, cAC, cCC)
    eatA_njit(C7, ND7, AC7, size, a0, alpha3, cAA, cAC)
    eatC_njit(C8, ND8, size, a0, alpha4, gamma, cAC, cCC)
    eatA_njit(C8, ND8, AC8, size, a0, alpha4, cAA, cAC)

    qenergy = 4
    add_queen_energy_njit(C1, ND1, AC1, size, qenergy)
    add_queen_energy_njit(C2, ND2, AC2, size, qenergy)
    add_queen_energy_njit(C3, ND3, AC3, size, qenergy)
    add_queen_energy_njit(C4, ND4, AC4, size, qenergy)
    add_queen_energy_njit(C5, ND5, AC5, size, qenergy)
    add_queen_energy_njit(C6, ND6, AC6, size, qenergy)
    add_queen_energy_njit(C7, ND7, AC7, size, qenergy)
    add_queen_energy_njit(C8, ND8, AC8, size, qenergy)

    maxclip = 3 + qenergy
    
    ND1 = np.clip(ND1 - ((C1 != 2) & (C1 != 3)) * maxclip, 0, maxclip)
    ND2 = np.clip(ND2 - ((C2 != 2) & (C2 != 3)) * maxclip, 0, maxclip)
    ND3 = np.clip(ND3 - ((C3 != 2) & (C3 != 3)) * maxclip, 0, maxclip)
    ND4 = np.clip(ND4 - ((C4 != 2) & (C4 != 3)) * maxclip, 0, maxclip)
    ND5 = np.clip(ND5 - ((C5 != 2) & (C5 != 3)) * maxclip, 0, maxclip)
    ND6 = np.clip(ND6 - ((C6 != 2) & (C6 != 3)) * maxclip, 0, maxclip)
    ND7 = np.clip(ND7 - ((C7 != 2) & (C7 != 3)) * maxclip, 0, maxclip)
    ND8 = np.clip(ND8 - ((C8 != 2) & (C8 != 3)) * maxclip, 0, maxclip)
    if tick%1==0:
        AC1 = AC1 + 1 - ((C1 != 4) & (C1 != 5)) * 1
        AC2 = AC2 + 1 - ((C2 != 4) & (C2 != 5)) * 1
        AC3 = AC3 + 1 - ((C3 != 4) & (C3 != 5)) * 1
        AC4 = AC4 + 1 - ((C4 != 4) & (C4 != 5)) * 1
        AC5 = AC5 + 1 - ((C5 != 4) & (C5 != 5)) * 1
        AC6 = AC6 + 1 - ((C6 != 4) & (C6 != 5)) * 1
        AC7 = AC7 + 1 - ((C7 != 4) & (C7 != 5)) * 1
        AC8 = AC8 + 1 - ((C8 != 4) & (C8 != 5)) * 1

    ss = random.choice([64])

    if tick%2==0:
        off_y = 0
        off_x = 0
    else:
        off_y = ss//2
        off_x = ss//2

    tumble_tiles_parallel_njit(ND1, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC1, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND2, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC2, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND3, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC3, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND4, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC4, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND5, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC5, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND6, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC6, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND7, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC7, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(ND8, size, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC8, size, ss, off_y, off_x)

    if tick>0:
        framecount += 1

        out = np.zeros((2160, 1920, 3), dtype='int')

        panel_w = size
        panel_h = size
        gap_x = (1920 - panel_w * 4) // 5
        gap_y = (2160 - panel_h * 2) // 3

        x = [gap_x, gap_x + panel_w + gap_x, gap_x + (panel_w + gap_x) * 2, gap_x + (panel_w + gap_x) * 3]
        y = [gap_y, gap_y + panel_h + gap_y]

        # species palette (R, G, B)
        COL_Q = np.array([255, 224, 110], dtype='int')   # gold
        COL_N = np.array([80, 210, 255], dtype='int')    # cyan
        COL_D = np.array([55, 115, 255], dtype='int')    # blue
        COL_A = np.array([255, 150, 70], dtype='int')    # orange
        COL_C = np.array([230, 70, 180], dtype='int')    # magenta

        panels = [(C1, ND1, 1.0), (C2, ND2, 1.1), (C3, ND3, 1.3), (C4, ND4, 1.7),
                  (C5, ND5, 1.0), (C6, ND6, 1.1), (C7, ND7, 1.3), (C8, ND8, 1.7)]

        for idx, (Cg, NDg, alpha) in enumerate(panels):
            row = idx // 4
            col = idx % 4
            e = np.clip(NDg * 10, 0, 255).astype('int')
            frame = np.zeros((size, size, 3), dtype='int')
            frame[:, :, 0] = (e // 4).astype('int')
            frame[:, :, 1] = (e // 2).astype('int')
            frame[:, :, 2] = e.astype('int')
            for mask, colv in [(Cg == 1, COL_Q), (Cg == 2, COL_N),
                               (Cg == 3, COL_D), (Cg == 4, COL_A),
                               (Cg == 5, COL_C)]:
                frame[mask] = np.clip(frame[mask] * 0.3 + colv * 0.7, 0, 255)
            frame = np.clip(frame, 0, 255)
            out[y[row]:y[row]+size, x[col]:x[col]+size] = frame

        nN = np.sum(C1 == 2)
        nD = np.sum(C1 == 3)
        nA = np.sum(C1 == 4)
        nC = np.sum(C1 == 5)
        print(framecount, ",", nN, ",", nD, ",", nA, ",", nC)

        writer.append_data(np.array(out, dtype=np.uint8))

writer.close()