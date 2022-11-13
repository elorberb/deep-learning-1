import unittest
from ass1_code import *


class MyTestCase(unittest.TestCase):
    def test_initialize_parameters(self):
        params = initialize_parameters([2, 1, 2])
        net_len = 2
        print(params)
        self.assertEqual(len(params), net_len)

    def test_linear_forward(self):
        A, W, B = np.array([2, 2]), np.array([2, 2]), np.array([0])
        A, linear_cache = linear_forward(A, W, B)
        print(A)
        expected = 8  # perform 2*2 +2*2
        self.assertEqual(A, expected)

    def test_linear_activation_forward(self):
        A_prev, W, B = np.array([2, 2]), np.array([2, 2]), np.array([0])
        A, cache = linear_activation_forward(A_prev, W, B, activation="relu")
        print(A)
        expected = 8  # perform 2*2 +2*2 and then apply relu
        self.assertEqual(A, expected)

    def test_softmax(self):
        Z = np.array([[1], [0]])
        A, activation_cache = softmax(Z)
        expected = np.array([0.73105858, 0.26894142])
        print(A)
        self.assertEqual(np.round(A[0], 3), np.round(expected[0], 3))

    def test_relu(self):
        Z = np.array([[1], [-1]])
        A, activation_cache = relu(Z)
        expected = [[1], [0]]
        self.assertListEqual(expected, A.tolist())

    def test_l_model_forward(self):
        params = {1: [np.array([[1, 1]]), np.array([[0.]])], 2: [np.array([[1]]), np.array([[0.]])]}
        X = np.array([1, 1])
        print("X", X.shape)
        AL, caches = l_model_forward(X=X, parameters=params, use_batchnorm=False)
        self.assertEqual(AL, softmax([2])[0])

        # params = {1: [np.array([[1, 1]]), np.array([[0.]])], 2: [np.array([[1], [1]]), np.array([[0.], [0.]])]}
        # X = np.array([1, 1])
        # AL, caches = l_model_forward(X=X, parameters=params, use_batchnorm=False)
        # print(AL)
        # self.assertListEqual(AL.tolist()[0], [0.5])

    def test_apply_batchnorm(self):
        A = np.array([[1, 1]])
        A_norm = apply_batchnorm(A)
        self.assertListEqual(A_norm.tolist(), [[0.0, 0.0]])

    def test_compute_cost(self):
        AL = np.array([[0.5], [0.5]])
        Y = np.array([[1], [0]])
        expected = 0.6931471805599453
        self.assertEqual(np.round(expected,3 ), np.round(compute_cost(AL, Y), 3))

    def test_flow(self):
        params = initialize_parameters([3, 3, 3, 3])
        X = np.array([[1, 2, 3, 2, 3, 4], [1, 1, 1, 2, 3, 4], [2, 2, 2, 2, 3, 4]])
        print(X.shape)
        AL, caches = l_model_forward(X=X, parameters=params, use_batchnorm=False)
        print(np.round(AL,3))
        print(AL.shape)




if __name__ == '__main__':
    unittest.main()
