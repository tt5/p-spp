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

# Game Loop

promote queens:
If D surrounded by N -> Q

remove queens:
If Q surrounded by A or C -> A

birth:
where positive energy  
where no Q  
nN >= 3 -> D  
nN < 3 -> N

eat C
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

tumble ND
