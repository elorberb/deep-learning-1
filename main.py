from ass1_code import *
import tensorflow as tf

if __name__ == '__main__':
    data = tf.keras.datasets.mnist.load_data()
    X_train, X_test, y_train, y_test = data[0][0], data[1][0], data[0][1], data[1][1]
    # preprocess data
    X_train_p, X_eval, y_train_p, y_eval = split_preprocess_train_data(X_train, y_train)
    X_test, y_test = split_preprocess_test_data(X_test, y_test)

    # define parameters
    layers = [784, 20, 7, 5, 10]
    batch_size = 200
    learning_rate = 0.009
    number_iterations = 20000
    batch_norm = False

    create_NN(X_train_p, X_eval, y_train_p, y_eval, X_test, y_test, batch_norm, X_train, y_train, layers, learning_rate,
              number_iterations, batch_size, name="batch_norm_FALSE")

    batch_norm = True
    create_NN(X_train_p, X_eval, y_train_p, y_eval, X_test, y_test, batch_norm, X_train, y_train, layers, learning_rate,
              number_iterations, batch_size, name="batch_norm_TRUE")
