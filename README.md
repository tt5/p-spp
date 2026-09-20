# Parameter Graph and Chip-Firing

parameters (a0, alpha, gamma, cAA, cAC, cCC)
parameter graph — a graph whose nodes sit at fixed (x, y) positions in grid coordinates.
The per-cell parameter grids are rebuilt whenever the graph changes by interpolating the node parameters onto the grid: with the current k_nearest this is a Voronoi (nearest-neighbour) assignment, so each cell inherits the parameters of its closest node; with a larger k the same interpolation becomes inverse-square (p = 2) IDW blending over the k nearest nodes.

Chip-firing runs on the graph: every lending_interval ticks, each non-corner node whose local cell is type N (2) performs a lending move — its chip count decreases by its degree and each neighbor gains 1 chip.
Every corner_interval ticks the corner nodes self-regulate: if a corner's chips drop below 0 it borrows (chips += degree, neighbors -= 1); if a corner's chips exceed the current lend threshold it lends.

If any node moved, the parameter grids are recomputed.

## Edge dynamics

For each non-corner node: if it has exactly one neighbor, it moves to the midpoint between itself and that neighbor; if the resulting distance is below the current collapse distance, it is kicked away from that neighbor just enough to reach that distance (plus the current kick buffer), then it reconnects to all other nodes.
If it has more than one neighbor, it deletes the edge to the neighbor with the highest chip count, breaking ties by closest Euclidean distance.

## Mutation

On node deletion.
The node adopts the other nodes's full parameter set, keeps one randomly chosen parameter from its own previous value, and mutates one randomly chosen parameter by ±0.1.
