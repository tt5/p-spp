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
#size = 1024
#size = 640
#size = 512

C1 = np.zeros((size,size), dtype='int')
ND1 = np.zeros((size,size), dtype='int')
AC1 = np.zeros((size,size), dtype='int')

C2 = np.zeros((size,size), dtype='int')
ND2 = np.zeros((size,size), dtype='int')
AC2 = np.zeros((size,size), dtype='int')

C3 = np.zeros((size,size), dtype='int')
ND3 = np.zeros((size,size), dtype='int')
AC3 = np.zeros((size,size), dtype='int')

# 1 queen
# 2 N
# 3 D
# 4 A
# 5 C

a0 = 0.5
alpha1 = 1.5 # help from defectors
alpha2 = 1.3 # help from defectors
alpha3 = 1.0 # help from defectors
gamma = 1.0 # cooperation for C

# interference
cAA = 1.0
cAC = 1.0
cCC = 1.0

# initial conditions
#bsize = 256
#C[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 2
bsize = 4
C1[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C2[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C3[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3

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

framecount = 0
print("time,", "N,", "D,", "A,", "C")
for tick in range(1000):

    promote_queens_njit(C1, AC1, ND1, size)
    remove_queens_njit(C1, ND1, size)
    promote_queens_njit(C2, AC2, ND2, size)
    remove_queens_njit(C2, ND2, size)
    promote_queens_njit(C3, AC3, ND3, size)
    remove_queens_njit(C3, ND3, size)

    birthAC_njit(C1, AC1, size)
    birthND_njit(C1, ND1, size)
    birthAC_njit(C2, AC2, size)
    birthND_njit(C2, ND2, size)
    birthAC_njit(C3, AC3, size)
    birthND_njit(C3, ND3, size)
    eatC_njit(C1, ND1, size, a0, alpha1, gamma, cAC, cCC)
    eatA_njit(C1, ND1, AC1, size, a0, alpha1, cAA, cAC)
    eatC_njit(C2, ND2, size, a0, alpha2, gamma, cAC, cCC)
    eatA_njit(C2, ND2, AC2, size, a0, alpha2, cAA, cAC)
    eatC_njit(C3, ND3, size, a0, alpha3, gamma, cAC, cCC)
    eatA_njit(C3, ND3, AC3, size, a0, alpha3, cAA, cAC)

    qenergy = 4
    add_queen_energy_njit(C1, ND1, AC1, size, qenergy)
    add_queen_energy_njit(C2, ND2, AC2, size, qenergy)
    add_queen_energy_njit(C3, ND3, AC3, size, qenergy)

    maxclip = 3 + qenergy
    
    ND1 = np.clip(ND1 - ((C1 != 2) & (C1 != 3)) * maxclip, 0, maxclip)
    ND2 = np.clip(ND2 - ((C2 != 2) & (C2 != 3)) * maxclip, 0, maxclip)
    ND3 = np.clip(ND3 - ((C3 != 2) & (C3 != 3)) * maxclip, 0, maxclip)
    if tick%1==0:
        AC1 = AC1 + 1 - ((C1 != 4) & (C1 != 5)) * 1
        AC2 = AC2 + 1 - ((C2 != 4) & (C2 != 5)) * 1
        AC3 = AC3 + 1 - ((C3 != 4) & (C3 != 5)) * 1

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

    if tick>0:
        framecount += 1

        out = np.zeros((1080, 1920, 3), dtype='int')

        panel_w = size
        total_w = panel_w * 3
        gap = (1920 - total_w) // 4
        y0 = (1080 - size) // 2
        x0 = gap
        x1 = gap + panel_w + gap
        x2 = gap + panel_w * 2 + gap * 2

        # species palette (R, G, B)
        COL_Q = np.array([255, 224, 110], dtype='int')   # gold
        COL_N = np.array([80, 210, 255], dtype='int')    # cyan
        COL_D = np.array([55, 115, 255], dtype='int')    # blue
        COL_A = np.array([255, 150, 70], dtype='int')    # orange
        COL_C = np.array([230, 70, 180], dtype='int')    # magenta

        # --- Panel 1 (alpha=1.0) ---
        e = np.clip(ND1 * 10, 0, 255).astype('int')
        frame1 = np.zeros((size, size, 3), dtype='int')
        frame1[:, :, 0] = (e // 4).astype('int')
        frame1[:, :, 1] = (e // 2).astype('int')
        frame1[:, :, 2] = e.astype('int')
        for mask, col in [(C1 == 1, COL_Q), (C1 == 2, COL_N),
                          (C1 == 3, COL_D), (C1 == 4, COL_A),
                          (C1 == 5, COL_C)]:
            frame1[mask] = np.clip(frame1[mask] * 0.3 + col * 0.7, 0, 255)
        frame1 = np.clip(frame1, 0, 255)
        out[y0:y0+size, x0:x0+size] = frame1

        # --- Panel 2 (alpha=1.1) ---
        e = np.clip(ND2 * 10, 0, 255).astype('int')
        frame2 = np.zeros((size, size, 3), dtype='int')
        frame2[:, :, 0] = (e // 4).astype('int')
        frame2[:, :, 1] = (e // 2).astype('int')
        frame2[:, :, 2] = e.astype('int')
        for mask, col in [(C2 == 1, COL_Q), (C2 == 2, COL_N),
                          (C2 == 3, COL_D), (C2 == 4, COL_A),
                          (C2 == 5, COL_C)]:
            frame2[mask] = np.clip(frame2[mask] * 0.3 + col * 0.7, 0, 255)
        frame2 = np.clip(frame2, 0, 255)
        out[y0:y0+size, x1:x1+size] = frame2

        # --- Panel 3 (alpha=1.3) ---
        e = np.clip(ND3 * 10, 0, 255).astype('int')
        frame3 = np.zeros((size, size, 3), dtype='int')
        frame3[:, :, 0] = (e // 4).astype('int')
        frame3[:, :, 1] = (e // 2).astype('int')
        frame3[:, :, 2] = e.astype('int')
        for mask, col in [(C3 == 1, COL_Q), (C3 == 2, COL_N),
                          (C3 == 3, COL_D), (C3 == 4, COL_A),
                          (C3 == 5, COL_C)]:
            frame3[mask] = np.clip(frame3[mask] * 0.3 + col * 0.7, 0, 255)
        frame3 = np.clip(frame3, 0, 255)
        out[y0:y0+size, x2:x2+size] = frame3

        nN = np.sum(C1 == 2)
        nD = np.sum(C1 == 3)
        nA = np.sum(C1 == 4)
        nC = np.sum(C1 == 5)
        print(framecount, ",", nN, ",", nD, ",",  nA, ",",  nC)

        writer.append_data(np.array(out, dtype=np.uint8))

writer.close()
