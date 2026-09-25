
initial grid is all D

# each tick

promote_queens_njit:
checks D cells → Q if ≥4 N neighbors

```txt
 N      N
NDN -> NQN
 N      N
```

birthND_njit:
overwrite all non-queen cells → D if exactly 3 N neighbors, else N.
