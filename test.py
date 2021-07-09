import unittest
import kdegree

class TestKDegreeAlgorithms(unittest.TestCase):

    d_list = [
        [8,7,7,6,3,2,2,1],
        [3,2,2,2],
        [3,3,3,2,1],
        [4,3,3,2,1,1]
    ]

    results_k_2 = [
        [8,8,7,7,3,3,2,2],
        [3,3,2,2],
        [3,3,3,2,2],
        [4,4,3,3,1,1]
    ]

    def test_dp_k_2(self):
        for i in range(len(self.d_list)): 
            result = kdegree.dp_algorithm(self.d_list[i], 2)
            self.assertEqual(result, self.results_k_2[i])
    
    def test_greedy_k_2(self):
        for i in range(len(self.d_list)): 
            result = kdegree.greedy_algorithm(self.d_list[i], 2)
            self.assertEqual(result, self.results_k_2[i])

if __name__ == '__main__':
    unittest.main()
