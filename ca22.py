import sys
import random
import math
from numba import njit, prange
import networkx as nx
import imageio
import numpy as np
from scipy.spatial import cKDTree

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

def chip_degree(nodes, nid):
    return nodes.degree(nid)

def chip_lending_move(nodes, nid):
    deg = nodes.degree(nid)
    nodes.nodes[nid]['chips'] -= deg
    for nb in nodes.neighbors(nid):
        nodes.nodes[nb]['chips'] += 1

def chip_borrowing_move(nodes, nid):
    deg = nodes.degree(nid)
    nodes.nodes[nid]['chips'] += deg
    for nb in nodes.neighbors(nid):
        nodes.nodes[nb]['chips'] -= 1

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
    _, best_node = tree.query(np.column_stack([_cell_x, _cell_y]), k=1)

    out_a0.ravel()  [:] = node_params[best_node, 0]
    out_alpha.ravel()[:] = node_params[best_node, 1]
    out_gamma.ravel()[:] = node_params[best_node, 2]
    out_cAA.ravel()  [:] = node_params[best_node, 3]
    out_cAC.ravel()  [:] = node_params[best_node, 4]
    out_cCC.ravel()  [:] = node_params[best_node, 5]
    return

VIDEO_W, VIDEO_H = 1920, 1080
VIDEO_FPS = 60

video_out_path = "new_video.mp4"
writer = imageio.get_writer(
    video_out_path,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=None,
    pixelformat="yuv420p",
    macro_block_size=None,
)

size = 896

C = np.zeros((size,size), dtype='int')
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
C[1:3, 1:-1] = 4
C[1:-1, 1:3] = 4
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
    100: (size//4,     size//3),
    110: (3*size//4,   size//3),
    120: (size//2,     size//3),
    130: (size//4,     size//6),
    140: (size//2,     size//6),
    150: (3*size//4,   size//6),
}

# Video layout — constant across all frames.
OFFY = (VIDEO_H - size) // 2
OFFX = (VIDEO_W - size) // 2

# Pre-allocated render buffers (reused every frame). uint8 matches what
# imageio / PIL expect, avoids int64 traffic and per-frame astype.
out   = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
frame = np.zeros((size, size, 3), dtype=np.uint8)

# species palette — built once; indexed by cell type.
#   0 empty -> white, 1 queen -> gold, 2 N -> black,
#   3 D -> blue, 4 A -> orange, 5 C -> magenta
_COL_Q = np.array([255, 224, 110], dtype=np.uint8)   # gold
_COL_D = np.array([80, 210, 255], dtype=np.uint8)
_COL_N = np.array([0, 0, 0], dtype=np.uint8)
_COL_A = np.array([195, 140, 60], dtype=np.uint8)    # orange
_COL_C = np.array([230, 70, 180], dtype=np.uint8)    # magenta
_COL_E = np.array([255, 255, 255], dtype=np.uint8)
_COLORS = np.array([_COL_E, _COL_Q, _COL_N, _COL_D, _COL_A, _COL_C], dtype=np.uint8)

# Bresenham line drawing — Numba-JIT'd, alpha-blended into frame buffer, no PIL.
_LINE_COLOR_R = np.float32(0)
_LINE_COLOR_G = np.float32(145)
_LINE_COLOR_B = np.float32(155)
_LINE_ALPHA = np.float32(136 / 255.0)
_LINE_W = 4
_LINE_R = _LINE_W // 2
_LINE_R2 = _LINE_R * _LINE_R

@njit(fastmath=True)
def draw_line_njit(img, h, w, y0, x0, y1, x1):
    """Bresenham with circular brush of radius _LINE_R. Alpha-blended into RGB uint8 img."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        # brush
        for by in range(max(0, y0 - _LINE_R), min(h, y0 + _LINE_R + 1)):
            for bx in range(max(0, x0 - _LINE_R), min(w, x0 + _LINE_R + 1)):
                if (by - y0) * (by - y0) + (bx - x0) * (bx - x0) <= _LINE_R2:
                    a = _LINE_ALPHA
                    inv_a = 1.0 - a
                    idx = (by * w + bx) * 3
                    img[idx]     = np.uint8(np.round(img[idx]     * inv_a + _LINE_COLOR_R * a))
                    img[idx + 1] = np.uint8(np.round(img[idx + 1] * inv_a + _LINE_COLOR_G * a))
                    img[idx + 2] = np.uint8(np.round(img[idx + 2] * inv_a + _LINE_COLOR_B * a))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def draw_line(img, y0, x0, y1, x1):
    """Thin wrapper — flattens frame to uint8 view, calls JIT'd core."""
    h, w = img.shape[:2]
    flat = img.reshape(-1)
    draw_line_njit(flat, h, w, y0, x0, y1, x1)

framecount = 0
for tick in range(1000):

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

    # ---- chip-firing dynamics on the parameter graph ----
    if tick%3 == 0:
        for nid in list(nodes.nodes()):
            if nid < 4:
                continue
            nx_node = nodes.nodes[nid]
            x, y = int(nx_node['x']), int(nx_node['y'])
            if 0 <= y < size and 0 <= x < size and (C[y, x]) == 2:
                chip_lending_move(nodes, nid)

    # For each non-starting node: delete one edge; if only one
    # edge remains, move to the midpoint of that edge, then if the new
    # position is too close to any other node, kick away from the nearest
    # one (and reconnect to all).
    if tick >= 160 and tick % 5 == 0:
        moved = False
        for nid in list(nodes.nodes()):
            if nid < 4:
                continue
            neighbors = list(nodes.neighbors(nid))
            if len(neighbors) == 1:
                moved = True
                other = neighbors[0]
                nid_data = nodes.nodes[nid]
                other_data = nodes.nodes[other]
                # mutation
                if tick>300 and other >= 4:
                    param_names = ['a0', 'alpha', 'gamma', 'cAA', 'cAC', 'cCC']
                    old_params = {p: nid_data[p] for p in param_names}
                    for p in param_names:
                        nid_data[p] = other_data[p]
                    keep_param = random.choice(param_names)
                    nid_data[keep_param] = old_params[keep_param]
                    keep_param = random.choice(param_names)
                    nid_data[keep_param] = old_params[keep_param]

                    nid_data['a0'] += random.choice([-0.1, 0.1])
                    nid_data['a0'] = max(a0-0.1, min(nid_data['a0'], a0+0.1))
                    nid_data['alpha'] += random.choice([-0.1, 0.1])
                    nid_data['alpha'] = max(alpha-0.1 , min(nid_data['alpha'], alpha+0.1))
                    nid_data['gamma'] += random.choice([-0.1, 0.1])
                    nid_data['gamma'] = max(gamma-0.1, min(nid_data['gamma'], gamma+0.1))
                    nid_data['cAA'] += random.choice([-0.1, 0.1])
                    nid_data['cAA'] = max(cAA-0.1, min(nid_data['cAA'], cAA+0.1))
                    nid_data['cAC'] += random.choice([-0.1, 0.1])
                    nid_data['cAC'] = max(cAC-0.1, min(nid_data['cAC'], cAC+0.1))
                    nid_data['cCC'] += random.choice([-0.1, 0.1])
                    nid_data['cCC'] = max(cCC-0.1, min(nid_data['cCC'], cCC+0.1))
                # 1. move to midpoint
                nid_data['x'] = (nid_data['x'] + other_data['x']) / 2
                nid_data['y'] = (nid_data['y'] + other_data['y']) / 2
                # 2. if too close to the collapse target, kick away from it
                min_dist = 256.0
                kx = nid_data['x'] - other_data['x']
                ky = nid_data['y'] - other_data['y']
                d = math.sqrt(kx * kx + ky * ky) + 1e-6
                if d < min_dist:
                    ux = kx / d
                    uy = ky / d
                    kick = (min_dist - d) * 0.5 + 2.0
                    nid_data['x'] += ux * kick
                    nid_data['y'] += uy * kick
                for other_nid in nodes.nodes():
                    if other_nid != nid:
                        nodes.add_edge(nid, other_nid)
            elif len(neighbors) > 1:
                # delete the edge to the neighbor with the highest chip count;
                # ties broken by shortest distance to nid; parameter dynamics
                # only happen on a tie
                moved = True
                def edge_key(n):
                    dx = nodes.nodes[nid]['x'] - nodes.nodes[n]['x']
                    dy = nodes.nodes[nid]['y'] - nodes.nodes[n]['y']
                    return (nodes.nodes[n]['chips'], -(dx*dx + dy*dy))
                victim = max(neighbors, key=edge_key)
                nodes.remove_edge(nid, victim)
        if moved:
            compute_param_grids(nodes, size, a0_grid, alpha_grid, gamma_grid, cAA_grid, cAC_grid, cCC_grid)

    if tick >= 200 and tick % 4 == 0:
        for nid in range(4):
            #print(f"{nid} {nodes.nodes[nid]['chips']}")
            if nodes.nodes[nid]['chips'] < 0:
                chip_borrowing_move(nodes, nid)
            elif nodes.nodes[nid]['chips'] > 10:
                chip_lending_move(nodes, nid)

    energy = 4
    add_energy_njit(C, ND, size, energy)

    maxclip = 3 + energy

    ND[~((C == 2) | (C == 3))] = 0
    np.clip(ND, 0, maxclip, out=ND)

    tumble_tiles_parallel_njit(ND, size, 64, 0, 0)

    if tick%1==0 and tick>0:
        framecount += 1

        # Video frame is 1920x1080. Global grid is size x size (1024).
        # Crop the grid to a 1024x1024 square and letterbox it into the frame,
        # centered with black padding.


        np.take(_COLORS, C, axis=0, out=frame)
        # ---- overlay parameter-graph edges (numpy) ----
        for (u, v) in nodes.edges():
            if u < 4 or v < 4:
                continue
            draw_line(frame,
                      int(nodes.nodes[u]['y']), int(nodes.nodes[u]['x']),
                      int(nodes.nodes[v]['y']), int(nodes.nodes[v]['x']))

        # letterbox: write the square crop into the center of the 1920x1080 frame
        out[OFFY:OFFY+size, OFFX:OFFX+size, :] = frame

        #nN = np.sum(C == 2)
        #nD = np.sum(C == 3)
        #nA = np.sum(C == 4)
        #nC = np.sum(C == 5)
        #print(f"{framecount}, {nN}, {nD}, {nA}, {nC}")

        print(tick)

        ## Print parameters of each inner (non-corner) node each tick
        #for nid in sorted(nodes.nodes()):
        #    if nid < 4:
        #        continue
        #    nd = nodes.nodes[nid]
        #    print(f"node {nid} tick {tick} x={nd['x']:.2f} y={nd['y']:.2f} "
        #          f"a0={nd['a0']:.3f} alpha={nd['alpha']:.3f} gamma={nd['gamma']:.3f} "
        #          f"cAA={nd['cAA']:.3f} cAC={nd['cAC']:.3f} cCC={nd['cCC']:.3f} "
        #          f"chips={nd['chips']} degree={nodes.degree(nid)}")

                
        writer.append_data(out)

writer.close()
