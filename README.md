# Discrete Dynamical System

cellular automaton

## Predator-Prey

energy terrain:  
AC, ND

cell map:  
C (0 (empty), ..., 5)

pkill for A:

$$\frac{a_0 + \alpha n_{\text{D}}} {1 + c_{\text{AA}} n_{\text{A}} + c_{\text{AC}} n_{\text{C}}}$$

pkill for C:

$$\frac{a_0 + \alpha n_{\text{D}}} {1 + c_{\text{CC}} - \gamma n_C + c_{\text{AC}}  n_{\text{A}}}$$

## Parameters

size (>= 192)

$a_0$ base attack  
$\alpha$ defector help  
$\gamma$ C cooperation

interference:
$c_{\text{AA}}$,
$c_{\text{CC}}$,
$c_{\text{AC}}$

## Species

prey:  
2 (N, Normal)  
3 (D, Defector)  

predators:  
4 (A, Aggressive)  
5 (C, Cooperative)  

promoted (prey -> predator):  
1 (Q, queen)

## Growth

Queen birth:

```txt
 N
NQN
 N
```

Queen death:

```txt
   C/A
C/A C C/A
   C/A
```

predators:  
where energy is possitive and not occupied by prey or queen.  
nA > nC -> 4  
nA < nC -> 5  
tie:  
nN > 0 -> 5  
else -> 4

prey:  
where energy is possitive and not occupied by queen.  
nN >= 3 -> 3   
else -> 2

## Eat

### Cooperative

pkill, all N, -> 5

### Aggressive

pkill, all N  
else (no N):  
pkill, one D  
and if nC >= 3: one C  

-> 4

## Sandpile

Add 4 energy to ND where queens are.

Remove all energy from ND where no prey.

Add 1 energy to AC where predators.

tumble (64x64)
