```python
from matplotlib import pyplot as plt
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
```


```python
def promote_queens(C, size):
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

def birthND(C, size):
    for y in range(size):
        for x in range(size):
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
            if north == 2: nN += 1
            if south == 2: nN += 1
            if east == 2: nN += 1
            if west == 2: nN += 1
            if nN == 3:
                C[y, x] = 3
            else:
                if C[y, x] != 1:
                    C[y, x] = 2

```


```python
size = 32

C = np.zeros((size,size), dtype='int8')
C = C+3

```


```python
for tick in range(2):

    promote_queens(C, size)

    birthND(C, size)

```


```python
#plt.imshow(C, interpolation='nearest');plt.show()
print(C)
```

    [[2 3 2 3 2 2 3 2 2 2 3 2 2 2 3 2 2 2 3 2 2 2 3 2 2 2 3 2 2 2 3 2]
     [3 2 1 2 3 2 3 2 3 3 2 2 3 2 3 2 3 3 2 2 3 2 3 2 3 3 2 2 3 2 2 3]
     [2 1 2 2 2 3 2 1 2 3 3 3 2 3 2 1 2 3 3 3 2 3 2 1 2 3 3 3 2 2 2 2]
     [3 2 2 2 3 2 2 2 1 2 3 2 3 2 2 2 1 2 3 2 3 2 2 2 1 2 3 2 3 2 2 3]
     [2 3 2 3 2 2 2 1 2 2 3 3 2 3 2 1 2 2 3 3 2 3 2 1 2 2 3 3 2 2 2 2]
     [2 2 3 2 2 2 1 2 2 2 3 2 3 2 2 2 2 3 2 2 3 2 2 3 3 3 2 3 2 3 2 3]
     [3 3 2 2 2 1 2 2 2 2 3 3 2 3 2 1 2 3 2 1 2 2 2 3 2 3 3 2 3 2 2 2]
     [2 2 1 2 1 2 2 2 2 2 3 2 3 2 2 2 1 2 1 2 1 3 3 2 2 3 2 2 2 3 2 3]
     [2 3 2 1 2 2 2 2 2 2 3 3 2 3 2 1 2 2 2 1 2 3 2 2 2 3 3 3 2 2 2 2]
     [2 3 3 2 2 2 2 2 2 2 3 2 3 2 2 2 2 2 1 2 2 3 3 3 3 2 3 2 3 2 2 3]
     [3 2 3 3 3 3 3 3 3 3 2 2 2 3 2 1 2 1 2 2 2 3 2 3 2 2 3 3 2 2 2 2]
     [2 2 3 2 3 2 3 2 3 2 2 2 3 2 2 2 1 2 2 2 2 3 3 2 2 2 3 2 3 2 2 3]
     [2 3 2 3 2 3 2 3 2 3 2 3 2 2 2 1 2 2 2 2 2 3 2 2 2 2 3 3 2 2 2 2]
     [2 2 3 2 3 2 3 2 3 2 3 2 2 2 1 2 2 2 2 2 2 3 3 3 3 3 2 3 2 3 2 3]
     [3 3 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 3 2 3 2 3 3 2 3 2 2 2]
     [2 2 1 2 1 2 1 2 1 2 1 2 1 2 2 2 2 2 2 2 2 3 3 2 2 3 2 2 2 3 2 3]
     [2 3 2 1 2 2 2 1 2 2 2 1 2 2 2 2 2 2 2 2 2 3 2 2 2 3 3 3 2 2 2 2]
     [2 3 3 2 2 3 3 2 2 2 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 2 3 2 3 2 2 3]
     [3 2 3 3 3 2 2 1 2 1 2 2 2 2 2 2 2 2 2 2 2 3 2 3 2 2 3 3 2 2 2 2]
     [2 2 3 2 3 2 1 2 1 2 2 2 2 2 2 2 2 2 2 2 2 3 3 2 2 2 3 2 3 2 2 3]
     [2 3 2 3 2 3 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 2 2 2 3 3 2 2 2 2]
     [2 2 3 2 3 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 3 2 3 2 2 3]
     [3 3 2 2 2 2 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 2 2 2 2 2 3 3 2 2 2 2]
     [2 2 1 2 1 3 3 2 2 3 3 2 2 3 3 2 2 3 3 2 2 2 2 2 2 2 3 2 3 2 2 3]
     [2 3 2 1 2 3 2 2 2 3 2 2 2 3 2 2 2 3 2 2 2 2 2 2 2 2 3 3 2 2 2 2]
     [2 3 3 2 2 3 3 3 3 2 2 2 2 3 3 3 3 2 2 2 2 2 2 2 2 2 3 2 3 2 2 3]
     [3 2 3 3 3 2 3 2 3 3 3 3 3 2 3 2 3 3 3 3 3 3 3 3 3 3 2 2 2 2 2 2]
     [2 2 3 2 3 3 2 2 3 2 3 2 3 3 2 2 3 2 3 2 3 2 3 2 3 2 2 2 3 2 2 3]
     [2 3 2 3 2 2 3 2 2 3 2 3 2 2 3 2 2 3 2 3 2 3 2 3 2 3 2 3 2 3 3 2]
     [2 2 2 2 2 3 2 3 2 2 2 2 2 3 2 3 2 2 2 2 2 2 2 2 2 2 2 2 3 2 2 3]
     [3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 2 2]
     [2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 3 2 2]]



```python

```
