import unittest
from code import *


class MyTestCase(unittest.TestCase):
    def test_initialize_parameters(self):
        params = initialize_parameters([2,1,1])
        net_len = 2
        self.assertEqual(len(params), net_len)

    def test_linear_forward(self):
        A, W, B = np.array([2, 2]), np.array([2, 2]), np.array([0])
        A, linear_cache = linear_forward(A, W, B)
        expected = 8  # perform 2*2 +2*2
        self.assertEqual(A, expected)

    def test_linear_activation_forward(self):
        A_prev, W, B = np.array([2, 2]), np.array([2, 2]), np.array([0])
        A, cache = linear_activation_forward(A_prev, W, B, activation="relu")
        expected = 8  # perform 2*2 +2*2 and then apply relu
        self.assertEqual(A, expected)

    def test_softmax(self):
        Z = [1,0]
        A, activation_cache = softmax(Z)
        expected = np.array([0.73105858, 0.26894142])
        print(A[0])
        self.assertEqual(np.round(A[0], 3), np.round(expected[0], 3))





if __name__ == '__main__':
    unittest.main()
