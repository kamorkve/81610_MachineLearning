import unittest
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import matplotlib.pylab as plt


# data is stored in array x and array y
# x: shape = (number of arrays, number of features.
# y: shape = (number of features)

data1 = np.genfromtxt('data_breast_cancer.csv', delimiter=',')
x = data1[:, :-1]  # for all but last column
y = data1[:, -1]  # for last column


class NeuralNetwork:

    def __init__(self, input_dim: int, hidden_layer: bool = True) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """
        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25
        self.lr = 0.1
        self.epochs = 50
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        if hidden_layer:
            self.hidden = np.zeros(self.hidden_units)

            self.weights = np.random.rand(self.input_dim + 1, self.hidden_units) - 0.5
            self.output_weights = np.random.rand(self.hidden_units + 1) - 0.5
        else:
            self.output_weights = np.random.rand(self.input_dim + 1) - 0.5

    def load_data(self, x_train, y_train, x_test, y_test) -> None:
        """
        Loads data into the neural network.
        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self) -> None:
        """
        Trains the neural network based on the data given in load_data
        :return: None
        """
        bias = np.array([1])

        for y in range(self.epochs):
            for x in range(len(self.x_train)):
                example = self.x_train[x]
                answer = self.y_train[x]
                if self.hidden_layer:
                    output = self.predict(example)

                    # print('Hidden layer, iteration no. ', y)

                    # initializing the temporary input and hidden layer with a bias node.
                    bias = np.array([1])
                    input_layer = np.hstack((bias, example))
                    temp_hidden = np.hstack((bias, self.hidden))

                    # calculate the value of the error function
                    error = (answer - output)

                    in_output = np.matmul(temp_hidden, self.output_weights)
                    delta_output = deriv_in(in_output) * error

                    # calculating delta with matrix
                    temp_output = self.output_weights[1:]
                    in_hidden = np.matmul(np.transpose(self.weights), input_layer)
                    delta = deriv_in(in_hidden) * temp_output * delta_output

                    # updating output_weights
                    self.output_weights += self.lr * temp_hidden * delta_output

                    # updating internal weights
                    a_i = input_layer * self.lr
                    ones = np.zeros((31, 25)) + 1
                    self.weights += np.transpose((np.transpose(ones * delta)) * a_i)

                    if np.isnan(output):
                        print('weights: ', self.weights)
                        print('output weights: ', self.output_weights)

                else:
                    output = self.predict(example)
                    # print('No hidden. Iteration no. ', y)

                    # initializing the temporary input layer with a bias node.
                    bias = np.array([1])
                    input_layer = np.hstack((bias, example))

                    # calculate the value of the error function
                    error = (answer - output)

                    in_output = np.matmul(input_layer, self.output_weights)
                    delta_output = deriv_in(in_output) * error

                    # updating output_weights
                    # this is the only place something can be wrong I think
                    self.output_weights += self.lr * input_layer * delta_output
        pass

    def predict(self, x: np.ndarray) -> float:
        """
        Predicts the result of a scan instance.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        bias = np.array([1])

        input_layer = np.hstack((bias, x))
        if self.hidden_layer:

            # calculate the value of the nodes in the hidden layer
            # input layer has one node less than the weights, since the weights include the bias.

            self.hidden = sigmoid(np.matmul(np.transpose(self.weights), input_layer))
            # calculate the output based on the values from hidden.

            temp_hidden = np.hstack((bias, self.hidden))
            output = np.matmul(temp_hidden, self.output_weights)
            output = sigmoid(output)

        else:
            # calculate the output based on the values from input.
            output = np.matmul(input_layer, self.output_weights)
            output = sigmoid(output)

        return output


def sigmoid(x):
    return np.round(1 / (1 + np.exp(-x)), decimals=12)


def deriv_in(x):
    return np.round(sigmoid(x) * (1 - sigmoid(x)), decimals=12)


def get_accuracy():
    """
    Uses KFold to get an average accuracy of the model.
    :return: A string describing how many splits were used in k-fold and a float between 0 and 1 of the f1 score.
    """
    kf = KFold(n_splits=10, shuffle=True)
    avg_correct = 0
    cum_score = 0
    # x1 = x[0:10, :]
    # y1 = y[0:10]
    for train_index, test_index in kf.split(x):
        # print('TRAIN INDEX: ', train_index, '\n\n TEST INDEX: ', test_index)
        NN = NeuralNetwork(30, True)
        NN.load_data(x[train_index], y[train_index], x[test_index], y[test_index])
        NN.train()

        n = len(NN.y_test)
        correct = 0
        pred_list = []
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = NN.predict(NN.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            pred_list.append(round(float(pred)))
            correct += NN.y_test[i] == round(float(pred))
        avg_correct += correct / n
        cum_score += f1_score(pred_list, NN.y_test)
        print(f1_score(pred_list, NN.y_test))
    return 'Average f1 score over ', kf.n_splits, ' folds is: ', round(cum_score/kf.n_splits, 4)


def learning_curve():
    """
    Plots the learning curve of the model using 10 to 400 training instances and 169 test instances.
    :return: A graph depicting the learning curve.
    """
    x_test = x[400:, :]
    y_test = y[400:]

    score_dict = {0: 0}

    for j in range(10, 400):
        print('Train data size: ', j)
        x_train = x[0:j, :]
        y_train = y[0:j]
        NN = NeuralNetwork(30, True)
        NN.load_data(x_train, y_train, x_test, y_test)
        NN.train()
        n = len(NN.y_test)
        correct = 0
        pred_list = []
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = NN.predict(NN.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            pred_list.append(round(float(pred)))
            correct += NN.y_test[i] == round(float(pred))
        score_dict[j] = f1_score(pred_list, NN.y_test)

    plot_list = sorted(score_dict.items())
    x_cord, y_cord = zip(*plot_list)
    plt.plot(x_cord, y_cord)
    plt.yticks(np.arange(min(y_cord), max(y_cord) + 0.05, 0.05))
    plt.ylabel('F1 Score')
    plt.xlabel('Number of training data')
    plt.grid(True)
    plt.show()

    return score_dict


if __name__ == '__main__':
    print(get_accuracy())
    # print(learning_curve())

