import resonance_solver as rs

from typing import *
import unittest

class GeneralTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.resonancePiecesPerType = rs.readResonancePiecesPerType('resonance_per_type.csv')
        cls.piecesStatData: rs.StatsDatabase = rs.readInStatsFromFile('resonance_piece_values.csv')

    def getPieces(self, resLevel: int, resType: str):
        return rs.constructPieces(
            GeneralTests.resonancePiecesPerType,
            GeneralTests.piecesStatData,
            resLevel,
            resType,
        )

    def testSolutionToShortname(self):
        pieces = self.getPieces(5, 'Z')
        solution = [1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        want = "Z _ _ l _ s O t | _ _ _ _ _ _ _"
        got = rs.Solution(solution, pieces).to_shortname()
        self.assertEqual(want, got)

    def testShortNamesToSolution(self):
        pieces = self.getPieces(5, 'Z')
        chosen = ['Z', 'L', '2', 'A', 'A', 'O']
        solution = rs.Solution.to_solution(chosen, pieces).solution
        self.assertListEqual(
            solution,
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        )

    def testShortNamesToSolution_invalidChosenElements(self):
        pieces = self.getPieces(5, 'Z')
        # Too many 'A', we don't have enough pieces to cover it
        chosen = ['Z', 'A', 'A', 'A']
        solution = rs.Solution.to_solution(chosen, pieces)
        self.assertIsNone(solution)



if __name__ == '__main__':
    unittest.main()