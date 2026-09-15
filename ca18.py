import sys
import random
import math
from numba import njit, prange
import networkx as nx
import imageio
import numpy as np
from chipfiring import CFGraph, CFDivisor

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# Global grid size (must be defined before the parameter graph below).
size = 1920

# Parameter graph (networkx): 4 corner nodes, fully connected.
# Each node carries position + 6 parameters:
#   x, y, a0, alpha, gamma, cAA, cAC, cCC
# compute_param_grids reads node attributes to build the per-cell
# parameter field; a 5th node can be injected mid-run (tick 100).


VIDEO_W, VIDEO_H = 1920, 1080
VIDEO_FPS = 60

video_out_path = "new_video.mp4"
writer = imageio.get_writer(
    video_out_path,
    fps=VIDEO_FPS,
    codec="libx264",
    quality=None,
    pixelformat="yuv420p",
)

@njit
def tumble_njit(spile):
    for i in range(256):
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
    """Compute per-cell (a0, alpha, gamma, cAA, cAC, cCC) by averaging k nearest nodes.

    nodes: networkx graph where each node carries x, y, a0, alpha, gamma, cAA, cAC, cCC.
    """
    if len(nodes) == 0:
        return np.zeros((size, size, 6), dtype='float64')
    node_ids = list(nodes.nodes())
    node_xy = np.array([[nodes.nodes[n]['x'], nodes.nodes[n]['y']] for n in node_ids], dtype='float64')
    node_params = np.array([[nodes.nodes[n]['a0'], nodes.nodes[n]['alpha'], nodes.nodes[n]['gamma'],
                             nodes.nodes[n]['cAA'], nodes.nodes[n]['cAC'], nodes.nodes[n]['cCC']]
                            for n in node_ids], dtype='float64')
    cell_coords = np.stack(np.meshgrid(np.arange(size), np.arange(size), indexing='ij'), axis=-1).reshape(-1, 2).astype('float64')
    diffs = cell_coords[:, None, :] - node_xy[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    knn = min(k, len(nodes))
    knn_idx = np.argpartition(dists, knn, axis=1)[:, :knn]
    avg_params = node_params[knn_idx].mean(axis=1)
    return avg_params.reshape(size, size, 6)


VIEW_SIZE = 512
size = 1920

VIEW_ORIGIN = 0  # top-left corner of the 192x192 view within the 256x256 global grid

C = np.zeros((size,size), dtype='int')
ND = np.zeros((size,size), dtype='int')
AC = np.zeros((size,size), dtype='int')
last_change = np.full((size, size), -1, dtype=np.int64)

# 1 queen
# 2 N
# 3 D
# 4 A
# 5 C

# Parameter graph nodes: [x, y, a0, alpha, gamma, cAA, cAC, cCC]
# Default: 4 corners.

nodes = nx.complete_graph(4)

for i, (x, y) in enumerate([(50, 50), (size-150, 50), (50, size//2), (size-150, size//2)]):
    nodes.nodes[i].update({
        'x': x, 'y': y,
        'a0': 0.5, 'alpha': 1.0, 'gamma': 1.0,
        'cAA': 1.0, 'cAC': 1.0, 'cCC': 1.0,
    })

# Initial conditions
#bsize = size//4
#C[size//2-bsize-1:size//2+bsize-1, size//2-bsize-1:size//2+bsize-1] = 3
C = C+3
C[0:2, 0:-1] = 4
C[0:-1, 0:2] = 4
C[-2:-1, 0:-1] = 4
C[0:-1, -2:-1] = 4

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

# ---- Chip-firing on the parameter graph (via chipfiring library) ----
def build_cf_graph_from_nodes(nodes):
    node_list = list(nodes.nodes())
    vertex_names = set(str(n) for n in node_list)
    edges = []
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            edges.append((str(node_list[i]), str(node_list[j]), 1))
    return CFGraph(vertex_names, edges)

cf_graph = build_cf_graph_from_nodes(nodes)
cf_divisor = CFDivisor(cf_graph, [(str(n), 0) for n in nodes.nodes()])

framecount = 0
print("time,", "N,", "D,", "A,", "C")
for tick in range(20000):
    # ---- snapshot for rendering (global, before view extraction) ----
    nd_pre = ND.copy()
    ac_pre = AC.copy()

    # ---- extract view from global grids ----
    #view_origin_tick = VIEW_ORIGIN + (tick//4)%(size-VIEW_SIZE)
    #halfsteps = size//VIEW_SIZE + (size//VIEW_SIZE - 1)
    halfsteps_h = 3+3
    halfsteps_v = 2+1
    view_origin_tick = VIEW_ORIGIN + (((tick//2)//halfsteps_h)%halfsteps_v)*(VIEW_SIZE//2)
    vy0 = view_origin_tick
    vy1 = view_origin_tick + VIEW_SIZE
    vx0 = ((tick//2)%halfsteps_h)*(VIEW_SIZE//2)
    vx1 = vx0 + VIEW_SIZE
    C_view   = C[vy0:vy1, vx0:vx1].copy()
    C_view_pre = C_view.copy()
    ND_view  = ND[vy0:vy1, vx0:vx1].copy()
    AC_view  = AC[vy0:vy1, vx0:vx1].copy()
    a0_view    = a0_grid[vy0:vy1, vx0:vx1].copy()
    alpha_view = alpha_grid[vy0:vy1, vx0:vx1].copy()
    gamma_view = gamma_grid[vy0:vy1, vx0:vx1].copy()
    cAA_view   = cAA_grid[vy0:vy1, vx0:vx1].copy()
    cAC_view   = cAC_grid[vy0:vy1, vx0:vx1].copy()
    cCC_view   = cCC_grid[vy0:vy1, vx0:vx1].copy()

    # ---- dynamics on the view (runs at VIEW_SIZE) ----
    promote_queens_njit(C_view, AC_view, ND_view, VIEW_SIZE)
    remove_queens_njit(C_view, ND_view, VIEW_SIZE)

    birthAC_njit(C_view, AC_view, VIEW_SIZE)
    birthND_njit(C_view, ND_view, VIEW_SIZE)
    eatC_njit(C_view, VIEW_SIZE, a0_view, alpha_view, gamma_view, cAC_view, cCC_view)
    eatA_njit(C_view, VIEW_SIZE, a0_view, alpha_view, cAA_view, cAC_view)

    if tick == 100:
        for i in range(1):
            new_id = len(nodes)
            nodes.add_node(new_id, x=size//4, y=size//3,
                           a0=random.randint(5, 20)/10, alpha=random.randint(5, 20)/10, gamma=random.randint(5, 80)/10,
                           cAA=random.randint(5, 80)/10, cAC=random.randint(5, 80)/10, cCC=random.randint(5, 80)/10)
            for existing in range(new_id):
                nodes.add_edge(new_id, existing)
            old_degrees = {str(n): cf_divisor.get_degree(str(n)) for n in cf_graph.vertices}
            cf_graph = build_cf_graph_from_nodes(nodes)
            new_degrees = [(str(n), old_degrees.get(str(n), 0)) for n in nodes.nodes()]
            cf_divisor = CFDivisor(cf_graph, new_degrees)
            param_grid = compute_param_grids(nodes, size, k=k_nearest)
            a0_grid = param_grid[:, :, 0].copy()
            alpha_grid = param_grid[:, :, 1].copy()
            gamma_grid = param_grid[:, :, 2].copy()
            cAA_grid = param_grid[:, :, 3].copy()
            cAC_grid = param_grid[:, :, 4].copy()
            cCC_grid = param_grid[:, :, 5].copy()

    if tick == 110:
        new_id = len(nodes)
        nodes.add_node(new_id, x=3*size//4, y=size//3,
                       a0=random.randint(5, 20)/10, alpha=random.randint(5, 20)/10, gamma=random.randint(5, 80)/10,
                       cAA=random.randint(5, 80)/10, cAC=random.randint(5, 80)/10, cCC=random.randint(5, 80)/10)
        for existing in range(new_id):
            nodes.add_edge(new_id, existing)
        old_degrees = {str(n): cf_divisor.get_degree(str(n)) for n in cf_graph.vertices}
        cf_graph = build_cf_graph_from_nodes(nodes)
        new_degrees = [(str(n), old_degrees.get(str(n), 0)) for n in nodes.nodes()]
        cf_divisor = CFDivisor(cf_graph, new_degrees)
        param_grid = compute_param_grids(nodes, size, k=k_nearest)
        a0_grid = param_grid[:, :, 0].copy()
        alpha_grid = param_grid[:, :, 1].copy()
        gamma_grid = param_grid[:, :, 2].copy()
        cAA_grid = param_grid[:, :, 3].copy()
        cAC_grid = param_grid[:, :, 4].copy()
        cCC_grid = param_grid[:, :, 5].copy()

    if tick == 120:
        new_id = len(nodes)
        nodes.add_node(new_id, x=size//4, y=size//6,
                       a0=random.randint(5, 20)/10, alpha=random.randint(5, 20)/10, gamma=random.randint(5, 80)/10,
                       cAA=random.randint(5, 80)/10, cAC=random.randint(5, 80)/10, cCC=random.randint(5, 80)/10)
        for existing in range(new_id):
            nodes.add_edge(new_id, existing)
        old_degrees = {str(n): cf_divisor.get_degree(str(n)) for n in cf_graph.vertices}
        cf_graph = build_cf_graph_from_nodes(nodes)
        new_degrees = [(str(n), old_degrees.get(str(n), 0)) for n in nodes.nodes()]
        cf_divisor = CFDivisor(cf_graph, new_degrees)
        param_grid = compute_param_grids(nodes, size, k=k_nearest)
        a0_grid = param_grid[:, :, 0].copy()
        alpha_grid = param_grid[:, :, 1].copy()
        gamma_grid = param_grid[:, :, 2].copy()
        cAA_grid = param_grid[:, :, 3].copy()
        cAC_grid = param_grid[:, :, 4].copy()
        cCC_grid = param_grid[:, :, 5].copy()

    if tick == 130:
        new_id = len(nodes)
        nodes.add_node(new_id, x=3*size//4, y=size//6,
                       a0=random.randint(5, 20)/10, alpha=random.randint(5, 20)/10, gamma=random.randint(5, 80)/10,
                       cAA=random.randint(5, 80)/10, cAC=random.randint(5, 80)/10, cCC=random.randint(5, 80)/10)
        for existing in range(new_id):
            nodes.add_edge(new_id, existing)
        old_degrees = {str(n): cf_divisor.get_degree(str(n)) for n in cf_graph.vertices}
        cf_graph = build_cf_graph_from_nodes(nodes)
        new_degrees = [(str(n), old_degrees.get(str(n), 0)) for n in nodes.nodes()]
        cf_divisor = CFDivisor(cf_graph, new_degrees)
        param_grid = compute_param_grids(nodes, size, k=k_nearest)
        a0_grid = param_grid[:, :, 0].copy()
        alpha_grid = param_grid[:, :, 1].copy()
        gamma_grid = param_grid[:, :, 2].copy()
        cAA_grid = param_grid[:, :, 3].copy()
        cAC_grid = param_grid[:, :, 4].copy()
        cCC_grid = param_grid[:, :, 5].copy()

    if tick == 140:
        new_id = len(nodes)
        nodes.add_node(new_id, x=2*size//4, y=size//12,
                       a0=random.randint(5, 20)/10, alpha=random.randint(5, 20)/10, gamma=random.randint(5, 80)/10,
                       cAA=random.randint(5, 80)/10, cAC=random.randint(5, 80)/10, cCC=random.randint(5, 80)/10)
        for existing in range(new_id):
            nodes.add_edge(new_id, existing)
        old_degrees = {str(n): cf_divisor.get_degree(str(n)) for n in cf_graph.vertices}
        cf_graph = build_cf_graph_from_nodes(nodes)
        new_degrees = [(str(n), old_degrees.get(str(n), 0)) for n in nodes.nodes()]
        cf_divisor = CFDivisor(cf_graph, new_degrees)
        param_grid = compute_param_grids(nodes, size, k=k_nearest)
        a0_grid = param_grid[:, :, 0].copy()
        alpha_grid = param_grid[:, :, 1].copy()
        gamma_grid = param_grid[:, :, 2].copy()
        cAA_grid = param_grid[:, :, 3].copy()
        cAC_grid = param_grid[:, :, 4].copy()
        cCC_grid = param_grid[:, :, 5].copy()


    # ---- chip-firing dynamics on the parameter graph ----
    # Fire every node that sits on a aggressive cell (C == 4), but only if it has
    # positive chips.
    for v in cf_graph.vertices:
        nid = int(str(v))
        nx_node = nodes.nodes[nid]
        x, y = int(nx_node['x']), int(nx_node['y'])
        if 0 <= y < size and 0 <= x < size and C[y, x] == 4 and cf_divisor.get_degree(str(v)) > 0:
            cf_divisor.lending_move(str(v))

    # For each non-starting node: delete one edge; if only one
    # edge remains, move to the midpoint of that edge and reconnect to all.
    if tick >= 100 and tick % 130 == 0:
        for nid in list(nodes.nodes()):
            if nid < 4:
                continue
            neighbors = list(nodes.neighbors(nid))
            if len(neighbors) == 1:
                other = neighbors[0]
                nid_data = nodes.nodes[nid]
                other_data = nodes.nodes[other]
                nid_data['x'] = (nid_data['x'] + other_data['x']) / 2
                nid_data['y'] = (nid_data['y'] + other_data['y']) / 2
                for other_nid in nodes.nodes():
                    if other_nid != nid:
                        nodes.add_edge(nid, other_nid)
            elif len(neighbors) > 1:
                # delete the edge to the neighbor with the highest chip count;
                # ties broken by shortest distance to nid
                def edge_key(n):
                    dx = nodes.nodes[nid]['x'] - nodes.nodes[n]['x']
                    dy = nodes.nodes[nid]['y'] - nodes.nodes[n]['y']
                    return (cf_divisor.get_degree(str(n)), -(dx*dx + dy*dy))
                victim = max(neighbors, key=edge_key)
                nodes.remove_edge(nid, victim)
                # parameter dynamics: copy from victim, keep one old, mutate one
                param_names = ['a0', 'alpha', 'gamma', 'cAA', 'cAC', 'cCC']
                nid_data = nodes.nodes[nid]
                victim_data = nodes.nodes[victim]
                old_params = {p: nid_data[p] for p in param_names}
                for p in param_names:
                    nid_data[p] = victim_data[p]
                keep_param = random.choice(param_names)
                nid_data[keep_param] = old_params[keep_param]
                mutate_param = random.choice(param_names)
                nid_data[mutate_param] += random.choice([-0.1, 0.1])
                for p in param_names:
                    nid_data[p] = max(0.5, min(nid_data[p], 8.0))
        # rebuild chip-firing graph and recompute parameter grids
        old_degrees = {str(n): cf_divisor.get_degree(str(n)) for n in cf_graph.vertices}
        cf_graph = build_cf_graph_from_nodes(nodes)
        new_degrees = [(str(n), old_degrees.get(str(n), 0)) for n in nodes.nodes()]
        cf_divisor = CFDivisor(cf_graph, new_degrees)
        param_grid = compute_param_grids(nodes, size, k=k_nearest)
        a0_grid = param_grid[:, :, 0].copy()
        alpha_grid = param_grid[:, :, 1].copy()
        gamma_grid = param_grid[:, :, 2].copy()
        cAA_grid = param_grid[:, :, 3].copy()
        cAC_grid = param_grid[:, :, 4].copy()
        cCC_grid = param_grid[:, :, 5].copy()

    if tick >= 100 and tick % 4400 == 0:
        for nid in range(4):
            cf_divisor.lending_move(str(nid))

    qenergy = 4
    add_queen_energy_njit(C_view, ND_view, AC_view, VIEW_SIZE, qenergy)

    maxclip = 3 + qenergy

    ND_view = np.clip(ND_view - ((C_view != 2) & (C_view != 3)) * maxclip, 0, maxclip)
    AC_view = AC_view + 1 - ((C_view != 4) & (C_view != 5)) * 1

    ss = random.choice([64])

    if tick % 2 == 0:
        off_y = 0
        off_x = 0
    else:
        off_y = ss // 2
        off_x = ss // 2

    tumble_tiles_parallel_njit(ND_view, VIEW_SIZE, ss, off_y, off_x)
    tumble_tiles_parallel_njit(AC_view, VIEW_SIZE, ss, off_y, off_x)

    ND_view = np.clip(ND_view, 0, 32)
    AC_view = np.clip(ND_view, 0, 32)

    # ---- write view back into global grids ----
    C[vy0:vy1, vx0:vx1]  = C_view
    ND[vy0:vy1, vx0:vx1] = ND_view
    AC[vy0:vy1, vx0:vx1] = AC_view

    # record last_change for cells whose species changed during dynamics
    changed = (C_view != C_view_pre)
    last_change[vy0:vy1, vx0:vx1][changed] = tick

    # ---- age-based defector conversion ----
    # Any cell that has not changed in the last STALE_TICKS ticks becomes a defector (3),
    # with ND/AC energy zeroed for that cell.
    STALE_TICKS = 600
    stale_mask = last_change <= tick - STALE_TICKS
    if stale_mask.any():
        C[stale_mask] = 2
        ND[stale_mask] = 0
        AC[stale_mask] = 0
        last_change[stale_mask] = tick

    if tick%(5)==0:
        framecount += 1

        out = np.zeros((1080, 1920, 3), dtype='int')

        # Fixed top-left crop: render the top 1080x1920 of the 1920x1920 global grid
        CROP_H = 1080
        CROP_W = 1920

        # ND background from cropped region
        e = np.clip(ND[:CROP_H, :CROP_W] * 10, 0, 255).astype('int')
        frame = np.zeros((CROP_H, CROP_W, 3), dtype='int')
        frame[:, :, 0] = (e // 4).astype('int')
        frame[:, :, 1] = (e // 2).astype('int')
        frame[:, :, 2] = e.astype('int')

        # species palette (R, G, B)
        COL_Q = np.array([255, 224, 110], dtype='int')   # gold
        COL_D = np.array([80, 210, 255], dtype='int')
        COL_N = np.array([0, 0, 0], dtype='int')
        COL_A = np.array([255, 180, 100], dtype='int')    # orange
        COL_C = np.array([230, 70, 180], dtype='int')    # magenta

        # overlay species colors (bright, mixed over background)
        C_crop = C[:CROP_H, :CROP_W]
        for mask, col in [(C_crop == 1, COL_Q), (C_crop == 2, COL_N),
                          (C_crop == 3, COL_D), (C_crop == 4, COL_A),
                          (C_crop == 5, COL_C)]:
            frame[mask] = np.clip(frame[mask] * 0.3 + col * 0.7, 0, 255)

        # faint motion trail from previous ND frame (soft)
        trail = np.clip(nd_pre[:CROP_H, :CROP_W] * 6, 0, 255).astype('int')
        frame[:, :, 0] = (frame[:, :, 0] * 0.8 + trail * 0.2).astype('int')
        frame[:, :, 1] = (frame[:, :, 1] * 0.8 + trail * 0.2).astype('int')
        frame[:, :, 2] = (frame[:, :, 2] * 0.8 + trail * 0.2).astype('int')

        frame = np.clip(frame, 0, 255)

        # ---- overlay parameter-graph edges (PIL) ----
        # node (x,y) maps 1:1 to pixel (x,y) in the 1920x1080 crop of the
        # 1920x1920 global grid. PIL clips lines to the image bounds, so
        # no manual endpoint clamping is needed.
        try:
            from PIL import Image, ImageDraw
            img = Image.fromarray(frame.astype(np.uint8))
            draw = ImageDraw.Draw(img)
            EDGE_COLOR = (10, 245, 255)
            EDGE_WIDTH = 2
            for (u, v) in nodes.edges():
                xu, yu = int(nodes.nodes[u]['x']), int(nodes.nodes[u]['y'])
                xv, yv = int(nodes.nodes[v]['x']), int(nodes.nodes[v]['y'])
                draw.line([(xu, yu), (xv, yv)], fill=EDGE_COLOR, width=EDGE_WIDTH)
            frame = np.array(img)
        except Exception as exc:
            print(f"[graph-overlay skipped] {exc}")

        out[:, :, :] = frame
        
        frame = out

        nN = np.sum(C == 2)
        nD = np.sum(C == 3)
        nA = np.sum(C == 4)
        nC = np.sum(C == 5)
        print(framecount, ",", nN, ",", nD, ",", nA, ",", nC)

        writer.append_data(frame.astype(np.uint8))

writer.close()
