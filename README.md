# Sandpile Cellular Automaton

E, 0, empty  
Q, 1, queen  
N, 2, normal prey  
D, 3, defector prey  
A, 4, aggressive predator  
C, 5, cooperative predator

cell matrix  
ND, energy matrix  
parameter matrices  

pkillA:

$$\frac{a_0 + \alpha n_{\text{D}}} {1 + c_{\text{AA}} n_{\text{A}} + c_{\text{AC}} n_{\text{C}}}$$

pkillC:

$$\frac{a_0 + \alpha n_{\text{D}}} {1 + c_{\text{CC}} - \gamma n_C + c_{\text{AC}}  n_{\text{A}}}$$

pkill threshold 0.5

eating: 8 cell neighborhood
birth: Von-Neumann (4) neighborhood

# Game Loop

promote queens:
If D surrounded by N -> Q

remove queens:
If Q surrounded by A or surrounded by C -> A

birth:
where positive energy  
where no Q  
nN >= 3 -> D  
nN < 3 -> N

eat C:
pkillC N

eat A:
pkillA N  
if no N eat all D  
if no N or D eat all C

fire and move parameter graph

add energy:
+4 where D

mask ND:
keep only energy where prey

tumble ND (tiles of 64x64)

# Parameter Graph and Chip-Firing

The spatial parameters (a0, alpha, gamma, cAA, cAC, cCC) are carried by a parameter graph — a fully-connected networkx graph whose nodes sit at fixed (x, y) positions in grid coordinates. Each node stores one value per parameter. The per-cell parameter grids (a0_grid, alpha_grid, gamma_grid, cAA_grid, cAC_grid, cCC_grid) are rebuilt whenever the graph changes by interpolating the node parameters onto the 1024x1024 grid: with k_nearest = 1 this is a Voronoi (nearest-neighbour) assignment, so each cell inherits the parameters of its closest node; with k >= 2 it would be inverse-square (p = 2) IDW blending over the k nearest nodes.

Four corner nodes are placed at (0,0), (size-1,0), (0,size-1), (size-1,size-1) and start with a0=0.5, alpha=1.0, gamma=1.0, cAA=cAC=cCC=1.0, chips=0. Starting at tick 100, new nodes are injected at ticks 100, 110, 120, 130, 140, 150 at various grid positions with random parameters: a0 in [0.5, 2.0], alpha in [0.5, 2.0], gamma in [0.5, 8.0], cAA/cAC/cCC in [0.5, 8.0]; each new node connects to all existing nodes and starts with chips = 0. After injection the parameter grids are recomputed.

Chip-firing runs on the graph: every 3 ticks, each non-corner node whose local cell is type N (2) performs a lending move — its chip count decreases by its degree and each neighbor gains 1 chip. Starting at tick 200, every 4 ticks the four corner nodes self-regulate: if a corner's chips drop below 0 it borrows (chips += degree, neighbors -= 1); if a corner's chips exceed 10 it lends.

Edge dynamics run every 5 ticks from tick 160 onward. For each non-corner node: if it has exactly one neighbor, it moves to the midpoint between itself and that neighbor; if the resulting distance is below 256, it is kicked away from that neighbor just enough to reach 256 (plus a small 2.0 buffer), then it reconnects to all other nodes. If it has more than one neighbor, it deletes the edge to the neighbor with the highest chip count, breaking ties by closest Euclidean distance. When a tie occurs (another neighbor shares the victim's chip count and distance key), parameter dynamics fire: the node adopts the victim's full parameter set, keeps one randomly chosen parameter from its own previous value, and mutates one randomly chosen parameter by ±0.1; all six parameters are then clamped to [0.5, 8.0]. If any node moved, the parameter grids are recomputed.
