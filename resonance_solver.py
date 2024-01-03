import csv
import numpy as np
import math
import time
import functools
import multiprocessing as mp

from typing import *
from collections import namedtuple
from collections import defaultdict
from pprint import pprint

from dlx_python import dlx
from criteria_config import *

GRID_SIZES = {
    1: (4,4),
    2: (4,4),
    3: (4,5),
    4: (4,5),
    5: (5,5),
    6: (5,5),
    7: (5,6),
    8: (5,6),
    9: (6,6),
    10: (7,7),
    11: (7,7),
    12: (7,7),
    13: (7,7),
    14: (7,7),
    15: (7,7),
}

PIECES_DIMENSIONS = {
    'T' :[       # T
        [0,1,0],
        [0,1,0],
        [1,1,1],
    ],
    'U' :[       # U
        [1,0,1],
        [1,1,1]
    ],
    'Z' :[       # Z
        [1,1,0],
        [0,1,0],
        [0,1,1]
    ],
    'Plus' :[    # +
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ],

    'L' :[        # L
        [1,0],
        [1,0],
        [1,1],
    ],
    'ReverseL' :[  # l
        [0,1],
        [0,1],
        [1,1]
    ],
    'S': [        # S
        [0,1,1],
        [1,1,0]
    ],
    'ReverseS': [ # s
        [1,1,0],
        [0,1,1]
    ],

    'Square': [ # O
        [1,1],
        [1,1]
    ],
    'LittleT': [ # t
        [0,1,0],
        [1,1,1]
    ],
    'Line': [  # |
        [1,1,1,1]
    ],
    'Corner': [ # C
        [1,1],
        [0,1]
    ],

    '2Piece': [  # 2
        [1,1]
    ],
    'DotAtk': [  # A
        [1]
    ],
    'DotDef': [  # D
        [1]
    ]
}

class Shape:
    def __init__(self, name, short_name, shape, num_unique_rots):
        self.name = name
        self.short_name = short_name
        self.shape = shape
        self.num_unique_rotations = num_unique_rots
        self.size = np.sum(shape)

        if num_unique_rots == 1:
            self.rots = {0: self.shape}
        elif num_unique_rots == 2:
            self.rots = {
                0: self.shape,
                90: np.rot90(self.shape, 1)
            }
        else:
            self.rots = {}
            for i in range(num_unique_rots):
                self.rots[i*90] = np.rot90(self.shape, i)

    def __str__(self):
        return "Shape(" + str(self.shape) + ")"

    def __repr__(self):
        return "Shape(" + str(self.shape) + ")"
            
SHAPES = {
    'T' : Shape('T', 'T', PIECES_DIMENSIONS['T'], 4),
    'U' : Shape('U', 'U', PIECES_DIMENSIONS['U'], 4),
    'Z' : Shape('Z', 'Z', PIECES_DIMENSIONS['Z'], 4),
    'Plus' : Shape('Plus', '+', PIECES_DIMENSIONS['U'], 4),
    'L' : Shape('L', 'L', PIECES_DIMENSIONS['L'], 4),
    'ReverseL' : Shape('ReverseL', 'l', PIECES_DIMENSIONS['ReverseL'], 4),
    'S': Shape('S', 'S', PIECES_DIMENSIONS['S'], 4),
    'ReverseS': Shape('ReverseS', 's', PIECES_DIMENSIONS['ReverseS'], 4),
    'Square': Shape('Square', 'O', PIECES_DIMENSIONS['Square'], 1),
    'LittleT': Shape('LittleT', 't', PIECES_DIMENSIONS['LittleT'], 4),
    'Line': Shape('Line', '|', PIECES_DIMENSIONS['Line'], 2),
    'Corner': Shape('Corner', 'C', PIECES_DIMENSIONS['Corner'], 4),
    '2Piece': Shape('2Piece', '2', PIECES_DIMENSIONS['2Piece'], 2),
    'DotAtk': Shape('DotAtk', 'A', PIECES_DIMENSIONS['DotAtk'], 1),
    'DotDef': Shape('DotDef', 'D', PIECES_DIMENSIONS['DotDef'], 1),
}

class Stat:
    def __init__(self, row):
        self.hp = 0
        self.atk = 0
        self.reality_def = 0
        self.mental_def = 0
        self.crit_rate = 0
        self.crit_rate_def = 0
        self.crit_dmg = 0
        self.crit_dmg_def = 0
        self.dmg_bonus = 0
        self.dmg_taken_reduction = 0

        if row is not None:
            self.hp = row['hp']
            self.atk = row['atk']
            self.reality_def = row['reality_def']
            self.mental_def = row['mental_def']
            self.crit_rate = row['crit_rate']
            self.crit_rate_def = row['crit_rate_def']
            self.crit_dmg = row['crit_dmg']
            self.crit_dmg_def = row['crit_dmg_def']
            self.dmg_bonus = row['dmg_bonus']
            self.dmg_taken_reduction = row['dmg_taken_reduction']

    def print_header(self):
        print("hp, atk, reality_def, mental_def, crit_rate, crit_dmg, crit_rate_def, crit_dmg_def, dmg_bonus, dmg_reduction")

    def __hash__(self):
        return hash((
            self.hp, self.atk, self.reality_def, self.mental_def,
            self.crit_rate, self.crit_rate_def,
            self.crit_dmg, self.crit_dmg_def,
            self.dmg_bonus, self.dmg_taken_reduction
        ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str([
            self.hp, self.atk, self.reality_def, self.mental_def, "|",
            self.crit_rate, self.crit_dmg, self.crit_rate_def, self.crit_dmg_def, "|",
            self.dmg_bonus, self.dmg_taken_reduction, "|",
            int(self.total_stat_gains())
        ])
    
    def __repr__(self):
        return str([
            self.hp, self.atk, self.reality_def, self.mental_def, "|",
            self.crit_rate, self.crit_dmg, self.crit_rate_def, self.crit_dmg_def, "|",
            self.dmg_bonus, self.dmg_taken_reduction, "|",
            int(self.total_stat_gains())
        ])
    
    def total_stat_gains(self):
        return (self.atk + self.dmg_bonus + 
                self.hp + self.reality_def + self.mental_def + self.dmg_taken_reduction + 
                self.crit_dmg + self.crit_rate + 
                self.crit_dmg_def + self.crit_rate_def)
        
    def __add__(self, other):
        stat = Stat(None)
        stat.hp = self.hp + other.hp
        stat.atk = self.atk + other.atk
        stat.reality_def = self.reality_def + other.reality_def
        stat.mental_def = self.mental_def + other.mental_def
        stat.crit_rate = self.crit_rate + other.crit_rate
        stat.crit_rate_def = self.crit_rate_def + other.crit_rate_def
        stat.crit_dmg = self.crit_dmg + other.crit_dmg
        stat.crit_dmg_def = self.crit_dmg_def + other.crit_dmg_def
        stat.dmg_bonus = self.dmg_bonus + other.dmg_bonus
        stat.dmg_taken_reduction = self.dmg_taken_reduction + other.dmg_taken_reduction
        return stat
    
    def __iadd__(self, other):
        self.hp += other.hp
        self.atk += other.atk
        self.reality_def += other.reality_def
        self.mental_def += other.mental_def
        self.crit_rate += other.crit_rate
        self.crit_rate_def += other.crit_rate_def
        self.crit_dmg += other.crit_dmg
        self.crit_dmg_def += other.crit_dmg_def
        self.dmg_bonus += other.dmg_bonus
        self.dmg_taken_reduction += other.dmg_taken_reduction
        return self
    
    def __sub__(self, other):
        stat = Stat(None)
        stat.hp = self.hp - other.hp
        stat.atk = self.atk - other.atk
        stat.reality_def = self.reality_def - other.reality_def
        stat.mental_def = self.mental_def - other.mental_def
        stat.crit_rate = self.crit_rate - other.crit_rate
        stat.crit_rate_def = self.crit_rate_def - other.crit_rate_def
        stat.crit_dmg = self.crit_dmg - other.crit_dmg
        stat.crit_dmg_def = self.crit_dmg_def - other.crit_dmg_def
        stat.dmg_bonus = self.dmg_bonus - other.dmg_bonus
        stat.dmg_taken_reduction = self.dmg_taken_reduction - other.dmg_taken_reduction
        return stat


type StatsDatabase = Dict[str, Dict[int, Stat]]
def readInStatsFromFile(filepath) -> StatsDatabase:
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row.keys():
                if k == "Piece":
                    continue
                row[k] = row[k].strip("%")
                if len(row[k]):
                    row[k] = float(row[k])
                else:
                    row[k] = 0

            if row['Piece'] not in data:
                data[row['Piece']] = {}
            data[row['Piece']][row['Piece Level']] = Stat(row)

    return data

class Piece:
    def __init__(self, name, level, shape, rotation, stats: Stat):
        self.name : str = name
        self.level : int = level
        self._shape : Shape = shape
        self._rotation : int = rotation
        self.stats : Stat = stats
        
    def __str__(self):
        return str({
            'name': self.name,
            'stats': self.stats
        })
    
    def __repr__(self):
        return str((
            self.name,
            self.stats
        ))
    
    @property
    def short_name(self):
        return self._shape.short_name
    
    @property
    def size(self):
        return self._shape.size
    
    @property
    def shape(self):
        return self._shape.rots[self._rotation]
    
    def set_rotation(self, new_rot):
        self._rotation = new_rot
        return self
    
    @property
    def width(self):
        return len(self.shape[0])

    @property
    def height(self):
        return len(self.shape)
    
    def get_rotations(self):
        return [x for x in self._shape.rots.keys()]
    
    def get_positions(self):
        for (ri, row) in enumerate(self.shape):
            for (ci, val) in enumerate(row):
                if val == 1:
                    yield (ri, ci)

    def is_resonance_piece(self):
        return self.name in ('Z', 'T', 'U', 'Plus')

    
def readResonancePiecesPerType(filepath):
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            resonanceType = row['Resonance Type']
            if resonanceType not in data:
                data[resonanceType] = {}
            resonanceLevel = int(row['Resonance Level'])
            if row['Piece'] == 'ResonanceType':
                row['Piece'] = resonanceType
            
            num_pieces = row['Number Pieces']
            if num_pieces is None:
                num_pieces = 0
            if len(num_pieces) == 0:
                num_pieces = 0
            num_pieces = int(num_pieces)
            if num_pieces == 0:
                continue

            pieceLevel = int(row['Piece Level'])
            if resonanceLevel not in data[resonanceType]:
                data[resonanceType][resonanceLevel] = []

            data[resonanceType][resonanceLevel].append({
                'piece': row['Piece'],
                'piece_level': pieceLevel,
                'num_pieces': num_pieces
            })

    return data

def constructPieces(
        resonancePiecesPerType,
        piecesStatData: StatsDatabase,
        resonanceLevel,
        resonanceType
    ) -> list[Piece]:
    data = []

    for piece in resonancePiecesPerType[resonanceType][resonanceLevel]:
        num_pieces = piece['num_pieces']
        for _ in range(num_pieces):
            data.append(Piece(
                piece['piece'], 
                piece['piece_level'],
                SHAPES[piece['piece']],
                0,  # 0 degrees rotation (default orientation)
                piecesStatData[piece['piece']][piece['piece_level']]
            ))  
    return data

class Solution:
    def __init__(self, solution, pieces):
        self.solution = solution
        self.pieces = pieces
        self.stats = self.getStats()

    def __str__(self):
        return str(self.stats)
    
    def __repr__(self):
        return str(self.stats)
    
    def print(self):
        return "{}\n{}".format(self.to_shortname(), self.stats)
    
    def to_shortname(self):
        result = []
        for si,ok in enumerate(self.solution):
            if ok:
                piece = self.pieces[si]
                result.append(piece.short_name)
            else:
                result.append('_')
        return " ".join(result)
    
    def contains_resonance_piece(self):
        return self.solution[0] == 1

    @classmethod
    def to_solution(cls, chosen: List[str], pieces: List[Piece]):
        used = [False for _ in range(len(pieces))]
        solution = [0 for _ in range(len(pieces))]
        for piece_shortname in chosen:
            found = False
            for pi, piece in enumerate(pieces):
                if used[pi]:
                    continue
                if piece.short_name == piece_shortname:
                    used[pi] = True
                    found = True
                    solution[pi] = 1
                    break
            if not found:
                return None
        return Solution(solution, pieces)
    
    def getStats(self):
        stat = Stat(None)
        for pi,ok in enumerate(self.solution):
            if not ok:
                continue
            piece: Piece = self.pieces[pi]
            stat += piece.stats
        return stat
    
    def iterate(self, fn):
        total = None
        for pi, ok in enumerate(self.solution):
            if ok:
                piece: Piece = self.pieces[pi]
                val = fn(piece)
                if total is None:
                    if isinstance(val, tuple):
                        total = list(val)
                    else:
                        total = val
                else:
                    if isinstance(val, tuple):
                        for i,x in enumerate(val):
                            total[i] += x
                    else:
                        total += fn(piece)
        return total

class Grid:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = None
        self.num_filled = 0
        self.num_inserts = 0
        self.reset()

    @classmethod
    def from_solution(self, 
                     grid_size: Tuple[int, int], 
                     answer,
                     pieces: List[Piece]):
        grid = Grid(grid_size)
        for (piece_index, rotation, coord) in answer:
            piece = pieces[piece_index]
            piece = piece.set_rotation(rotation)
            grid.insert_piece(coord, piece)
        return grid

    def reset(self):
        self.grid = self.create_empty_grid(self.grid_size)
        self.num_filled = 0
        self.num_inserts = 0

    def create_empty_grid(self, grid_size):
        grid = []
        for row in range(grid_size[0]):
            grid.append([])
            for col in range(grid_size[1]):
                grid[row].append(None)
        return grid
    
    def insert_piece(self, coord, piece):
        self.num_inserts += 1
        for (ri, row) in enumerate(piece.shape):
            for (ci, col) in enumerate(row):
                if col == 0:
                    continue
                new_row = coord[0] + ri
                new_col = coord[1] + ci
                self.grid[new_row][new_col] = piece.short_name
        self.num_filled += piece.size

    def remove_piece(self, coord, piece):
        for (ri, row) in enumerate(piece.shape):
            for (ci, col) in enumerate(row):
                new_row = coord[0] + ri
                new_col = coord[1] + ci
                if col == 1:
                    self.grid[new_row][new_col] = None
        self.num_filled -= piece.size
                
    def will_fit(self, coord, piece):
        # Check for conflicting squares
        for (ri, row) in enumerate(piece.shape):
            for (ci, col) in enumerate(row):
                new_row = coord[0] + ri
                new_col = coord[1] + ci

                if col == 0:
                    continue
                if not(0 <= new_row < self.grid_size[0]):
                    return False
                if not(0 <= new_col < self.grid_size[1]):
                    return False
                if self.grid[new_row][new_col] is not None:
                    return False
        return True
    
    def is_full(self):
        return self.slots_left() == 0
    
    def slots_left(self):
        return self.grid_size[0]*self.grid_size[1] - self.num_filled

    @property
    def width(self):
        return self.grid_size[0]
    
    @property
    def height(self):
        return self.grid_size[1]
    
    def print_grid(self):
        line = ""
        for row in self.grid:
            for col in row:
                if col is None:
                    line += '. '
                else:
                    line += col + ' '
            line += "\n"
        return line
    
    def __str__(self):
        return self.print_grid()

    def __repr__(self):
        return self.print_grid()

def subset_sum(lower, upper, target, sizes, n):
    solutions = []
    for num in range(lower, upper+1):
        total = 0

        solution = [0 for _ in range(n)]
        for i in range(n):
            if num & (1 << i) > 0:
                total += sizes[n-i-1]
                solution[n-i-1] = 1
        
        if total == target:
            solutions.append(solution)

    return solutions

def subset_sum_multithread(pieces, target, num_threads=14):
    n = len(pieces)
    limit = int(math.pow(2, n))
    sizes = [pieces[i].size for i in range(n)]

    portion = (limit / num_threads)
    bounds = []
    lower = 0
    while lower < limit:
        upper = min(lower + portion, limit)
        bounds.append((int(lower), int(upper)))
        lower += portion + 1

    pool = mp.Pool()
    results = []
    for (lower, upper) in bounds:
        pid = pool.apply_async(subset_sum, (lower, upper, target, sizes, n))
        results.append(pid)
    
    solutions = []
    for res in results:
        solutions.extend(res.get())
    return solutions

def get_resonance_solutions(grid_size, pieces):
    want = grid_size[0] * grid_size[1]
    start = time.time()
    piece_subsets = subset_sum_multithread(pieces, want)
    end = time.time()
    # print("Piece subsets:", end-start, "secs")
    return piece_subsets

def order_resonance_solutions(
        solutions: List[List[int]],
        pieces: List[Piece],
        criteria: Criteria
    ):
    solutions = [Solution(x, pieces) for x in solutions]
    solutions: List[Solution] = [x for x in solutions if x.contains_resonance_piece()]
    def filter_equal_stats(solutions):
        solutions2 = []
        seen = set()
        for solution in solutions:
            if solution.stats in seen:
                continue
            seen.add(solution.stats)
            solutions2.append(solution)
        return solutions2
    solutions: List[Solution] = filter_equal_stats(solutions)
    return solutions

def get_solution_candidates(
        solutions: List[Solution],
        criteria: Criteria,
        top_n=3
    ):

    def partition_into_blocks(solutions, get_stat_fn, top_n):
        blocks = defaultdict(list)
        for solution in solutions:
            stat = get_stat_fn(solution)
            blocks[stat].append(solution)
        return {stat: blocks[stat] for stat in sorted(blocks, reverse=True)[:top_n]}
        
    def continue_partition(blocks, solutions_fn, get_stat_fn, top_n):
        new_blocks = {}
        for stat, blks in blocks.items():
            if isinstance(blks, list):            
                new_blocks[stat] = solutions_fn(blks, get_stat_fn, top_n)
            else:
                new_blocks[stat] = continue_partition(blks, solutions_fn, get_stat_fn, top_n)
        return new_blocks
    
    def skim_solution(solutions, get_stat_fn, top_n=1):
        return sorted(solutions, key=get_stat_fn, reverse=True)[:top_n]

    blocks = solutions
    for i,v in enumerate(criteria.get_ordering()):
        top_n, ordering_fn = v
        if i == 0:
            blocks = partition_into_blocks(blocks, ordering_fn, top_n)
        else:
            blocks = continue_partition(
                blocks, partition_into_blocks, ordering_fn, top_n
            )
    blocks = continue_partition(
        blocks, 
        skim_solution,
        lambda x: x.stats.total_stat_gains(),
        top_n
    )

    # pprint(blocks)

    final_solutions = []
    def extract_solutions(solutions, fn, top_n):
        final_solutions.extend(solutions)
    continue_partition(blocks, extract_solutions, lambda x: x, top_n)
    return final_solutions

def print_pieces(pieces):
    for i, piece in enumerate(pieces):
        print(i)
        pprint(piece)

class Solver:
    Answer = namedtuple('Answer', ['piece_index', 'rotation', 'coord'])
    
    def __init__(self, grid_size, pieces):
        self.grid = Grid(grid_size)
        self.pieces = pieces

    def _generateColumns(self, grid, pieces):
        # each column represent the set elements
        columns = []
        # positions in the grid
        for ri in range(grid.height):
            for ci in range(grid.width):
                columns.append((("coord", ri, ci), dlx.DLX.PRIMARY))

        # the pieces
        pieces_index = {}
        for pi, piece in enumerate(pieces):
            key = ("piece", pi, piece.short_name)
            value = len(columns)
            columns.append((key, dlx.DLX.PRIMARY))
            pieces_index[key] = value

        return (columns, pieces_index)
    
    def _generateRows(self, pieces, pieces_index):
        # each row is a constraint
        # each constraint is a placement of a piece on the grid, accounting
        # for rotations and the shape of the block
        rows = []
        row_names = []
        for pi,piece in enumerate(pieces):
            piece_key = ("piece", pi, piece.short_name)
            piece_column = pieces_index[piece_key]

            for rot in piece.get_rotations():
                piece = piece.set_rotation(rot)
                
                for ri in range(self.grid.height):
                    for ci in range(self.grid.width):
                        coord = (ri,ci)
                        if not self.grid.will_fit(coord, piece):
                            continue

                        row = [piece_column]
                        row_name = (piece_key,rot,(ri,ci))
                        for piece_pos in piece.get_positions():
                            nr, nc = ri + piece_pos[0], ci + piece_pos[1]
                            position_column = nr*self.grid.height + nc
                            row.append(position_column)
                        rows.append(row)
                        row_names.append(row_name)
        return (rows, row_names)
    
    def _convertSolutionIntoAnswer(self, solution, pieces_mapping, dlxObj):
        answer = []
        for rowId in solution:
            row_name = dlxObj.N[rowId]
            piece_index = row_name[0][1]
            piece_index = pieces_mapping[piece_index]
            rot = row_name[1]
            ri,ci = row_name[2]
            answer.append(Solver.Answer(piece_index, rot, (ri,ci)))
        return answer
    
    def _fill_grid(self, grid, answer, pieces):
        grid.reset()
        for (piece_index, rot, coord) in answer:
            piece = pieces[piece_index]
            piece = piece.set_rotation(rot)
            grid.insert_piece(coord, piece)
        return grid

    def solve(self, chosenPieces):
        sub_pieces = []
        # maps the index of sub_pices into to its original index in self.pieces
        # this is so that the returned answer can relate the piecse chosen from 
        # the solution directly to the pieces
        sub_pieces_mapping = {}  
        for i,ok in enumerate(chosenPieces):
            if ok:
                sub_pieces.append(self.pieces[i])
                sub_pieces_mapping[len(sub_pieces)-1] = i

        columns, pieces_index = self._generateColumns(self.grid, sub_pieces)
        rows, row_names = self._generateRows(sub_pieces, pieces_index)
        

        start_time = time.time()
        dlxObj = dlx.DLX(columns=columns, rows=rows, rowNames=row_names)
        solution = None
        for x in dlxObj.solve():
            solution = x
            break
        end_time = time.time()

        if solution:
            return self._convertSolutionIntoAnswer(solution, sub_pieces_mapping, dlxObj)
        else:
            return None


def main2():
    resonancePiecesPerType = readResonancePiecesPerType('resonance_per_type.csv')
    piecesStatData: StatsDatabase = readInStatsFromFile('resonance_piece_values.csv')

    # Example resonance builds from the googlesheet
    # https://docs.google.com/spreadsheets/d/12NQ9kxcL4Iz4ZdNbsN7iZCBxtf2qNQ20i1YNcfo6QSc/htmlview
    fromDocs = {
        ('Z',5, DefBuild())   : ['2','s','D','D','t','Z','O','|'],
        ('U',5, DefBuild())   : ['U','2','s','D','D','s','t','O'],
        ('Z',10, AttackBuild())  : ['A','A','|','L','Z','L','C','C','L','O','t','t','|','O'],
        ('U',9, AttackBuild())   : ['|','|','O','S','t','L','L','A','A','D','U'],
        ('+',5, AttackBuild())   : ['|','t','+','C','C','A','A','O'],
        ('U',10, CritBuild()) : ['O','O','t','U','|','|','C','C','t','C','2','l','l'],
        ('T',7, CritBuild())  : ['T','l','A','A','S','L','L','C'],
        ('T',10, CritBuild()) : ['L','|','S','S','S','C','C','T','l','C','t','L','A','2'],
        ('Z',10, CritBuild()) : ['Z', 'L', 'L', 'L', 'l', 'l', 'S', 'S', '|', 'C', 'C', 'O', '2'],
        ('+',10, CritBuild()) : ['L','C','D','O','S','O','C','A','A','C','l','l','L','L','+'],
    }

    for key in fromDocs:
        resonanceType, resonanceLevel, criteria = key
        grid_size = GRID_SIZES[resonanceLevel] 
        pieces = constructPieces(
            resonancePiecesPerType,
            piecesStatData,
            resonanceLevel,
            resonanceType
        )

        print(resonanceType, resonanceLevel, criteria)
        dlxSolver = Solver(grid_size, pieces)
        solutions: List[List[int]] = get_resonance_solutions(grid_size, pieces)
        solutions: List[Solution] = order_resonance_solutions(solutions, pieces, criteria)
        solutions: List[Solution] = get_solution_candidates(solutions, criteria, 2)

        answers: List[Solution] = []
        for solution in solutions:    
            answer = dlxSolver.solve(solution.solution)
            if answer is not None:
                answers.append((solution, answer))

        for (i,(solution, answer)) in enumerate(answers):
            print(solution.print())
            # print(Grid.from_solution(grid_size, answer, pieces))

        desiredSolution: Solution = Solution.to_solution(fromDocs[key], pieces)
        answer = dlxSolver.solve(desiredSolution.solution)
        print(desiredSolution.print())
        print(Grid.from_solution(grid_size, answer, pieces))

def main():
    resonancePiecesPerType = readResonancePiecesPerType('resonance_per_type.csv')
    piecesStatData: StatsDatabase = readInStatsFromFile('resonance_piece_values.csv')

    for resonanceLevel in range(2,3):   
        grid_size = GRID_SIZES[resonanceLevel] 
        for resonanceType in ('Z', 'T', 'U', 'Plus'):
            pieces = constructPieces(
                resonancePiecesPerType,
                piecesStatData,
                resonanceLevel,
                resonanceType
            )
            for criteria in [AttackBuild(), CritBuild(), DefBuild()]:
                print(f"Level: {resonanceLevel}, Type: {resonanceType}, {criteria}")

                dlxSolver = Solver(grid_size, pieces)
                solutions: List[List[int]] = get_resonance_solutions(grid_size, pieces)
                solutions: List[Solution] = order_resonance_solutions(solutions, pieces, criteria)
                solutions: List[Solution] = get_solution_candidates(solutions, criteria, 2)
                print("Number of candidates", len(solutions))

                answers: List[Solution] = []
                for solution in solutions:    
                    answer = dlxSolver.solve(solution.solution)
                    if answer is not None:
                        answers.append((solution, answer))

                for (i,(solution, answer)) in enumerate(answers):
                    solution.stats.print_header()
                    print(solution.print())
                    print(Grid.from_solution(grid_size, answer, pieces))

                dlxSolver = Solver(grid_size, pieces)

if __name__ == "__main__":
    main()