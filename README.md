# Reverse 1999 Resonance Solution Finder

This repository contains a program to generate all the possible solutions to the Resonance puzzle for any given resonance level (2-15), Resonance Type (Z,U,T,+) and any paritcular build (Crit build, etc).

# How To
```
git clone https://github.com/Stymphalian/r99-resonance-solver
cd r99-resonance-solver

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Note this can take several minutes to complete
python resonance_solver.py
```

The output:
```
hp, atk, reality_def, mental_def, crit_rate, crit_dmg, crit_rate_def, crit_dmg_def, dmg_bonus, dmg_reduction
Z _ L L _ _ S S _ O O t t | | _ _ C _ _ _ _ _ D
[27.58, 31.47, 26.48, 26.55, '|', 14.5, 6.0, 14.0, 0, '|', 13.0, 9.5, '|', 169]
Z Z L L O O |
L Z D L O O |
L Z Z L S S |
L L t S S S |
t t t O O S S
t t t O O C S
t | | | | C C

...
```

The `Z _ L L _ ...` tells you which pieces were selected for this solution. The `_` are pieces which were not included. \
The `[27.58, 31.47, 26.48, ...]` tells you the stats gained from this particular solution. It is a sum of all the stats from all the pieces of the solution. The `hp, atk, reality_def,...` header on the first line tells you which column corresponds to which stat. \
Finally the printed 7x7 grid of letters is the specific placements of each piece in the grid.


# How It Works
The general program flow is this: 

0. Read in all the Pieces and Stats data
1. For a given resonance level and resonance type
2. Generate all the sets of pieces which _could_ be a solution to the puzzle
3. Sort the list of solutions based on a set of criteria in order to maximize a particular set of stats (i.e. Crit oriented, Atk focus, etc). Take the top `N` solutions as candidates to try and solve the puzzle.
4. Run the DLX algorithm to find a valid arrangement of the pieces on the board. 
5. Print out the Answer into the command line.

# Informational
There are a few parts to the solver.
1. Solver/dlx_python
2. Shape/Stat/Piece/Grid/Solution
3. CSV Files
3. Criteria

## Solver and DLX
Firstly, the `Solver` is the main class which actually solves the puzzle by trying to place the pieces into the `N x N` puzzle board. After a bit of online searching you will discover that the Reverse 1999 Resonance puzzle is just trying to place `polyomino` pieces in a grid. This is a known problem and it is shown that it can be solved by reducing it into an exact cover problem. Exact cover is an NP-hard problem but gladly with only 26 pieces at R15 and a 7x7 grid it is still solvable quite quickly. A common solution to this problem is to use Donald Knuth's Algorithm X which uses the Dancing Links programming technique (i.e DLX - Dancing Links X). I found a good python implementation of it [here](https://github.com/sraaphorst/dlx-python) by Sebastian Raaphorst. It is copied directly into this repository under the `dlx_python` folder.

## Shape/Stat/Piece/Grid/Solution
I heavily use an object oriented programming style in this program. There are classes for each of the concepts of this problem. 

| Class | Description |
|-------| ------------|
| Shape | This contains the data for all the shapes which are available. Check out `PIECES_DIMENSIONS` to see shapes as well as the names I assigned to them. The shapes available are `T, U, Z, Plus, L, ReverseL, S, ReverseS, Square, LittleT, Line, Corner, 2Piece, DotAtk, DotDef`
| Piece | This class encapsulates the concept of a piece that is used in the puzzle. Each piece has a name, level, rotation (in degrees), and stats. The rotation directly affects the `shape` of the piece. It should be set to 0,90,180 or 270 degrees. We try to fit these pieces into the `Grid`
| Stat | Every resonance piece at a given level has a set of stats associated with it (ie. `hp, atk, mental_def, crit_rate, etc`). This class just holds all this data for easy access and to help with debugging/printing.
| Grid | A convenient class for construcing the `N x N` board and inserting/removing pieces in the grid. Has some helper methods for checking if a piece "fits" at a certain location and doesn't overlap another placed piece. Mainly used so that we can print out a human-readable grid to the console.
| Solution | A solution is a set of pieces which will directly fit into the grid/puzzle board. Internally it is represented as a bit-array. A `1` at position `i` tells us that piece `pieces[i]` should be included in this final solution.

## CSV Files
`resonance_per_type.csv`  \
There are only 4 major Resonance Types which are identifiable by the single large resonance piece shaped like a `T, U, Z and Plus`. At each resonance level you are given a set of pieces and each piece has there own level. This CSV files lists out the pieces which are available for use at each resonance level for each Resonance Type.

`resonance_piece_values.csv`  \
Every piece has some given stats associated with them. In total there are: \
`hp, atk, reality_def, mental_def, crit_rate, crit_dmg` \
`crit_rate_def, crit_dmg_def, dmg_bonus, dmg_reduction`

This CSV files list out the percent of the stat gained for each piece at each level. Note that the Resonance Type (i.e Z, U, T, +) are given in percents for the `hp,atk,reality_def,mental_def` stats. In-game the UI presents these stats as raw stat increases (i.e +45 ATK) but if you do some digging and look at all the pieces across all the Resonance Types,levels and characters you will see that they are in-fact just a percentage of the Character's max stat. `Z,T and Plus` all share the same stat progression. Only `U` Resonance Type is different in terms of the stat gains.

## Criteria 
The `criteria_config.py` file contains a few simple classes. It is here where we define the `AtkBuild`,`CritBuild` and `DefBuild` classes which are used to determine which `stats` gained from a `Solution` should be prioritized. Definining these criteria allows us to do a deterministic ordering of the list of solutions so that we only show the most relevant resonance builds which maximize the stats that we want.

For example: \
The `CritBuild` prioritizes maximizing `crit_rate` over any other stat. Secondarily it will try to maximize `atk` and `crit_dmg`, and finally if all else is equal it will try to maximize `def`

If you want to find your own solutions to maximize some certain stat gains you should subclass your own `Criteria` class and define your own `ordering`.
