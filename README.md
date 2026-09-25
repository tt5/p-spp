
initial grid is all D

# each tick

## promote queens

checks D cells → Q if 4 N direct neighbors

```txt
 N      N
NDN -> NQN
 N      N
```

## birthND

overwrite all non-queen cells → D if exactly 3 N neighbors, else N.
