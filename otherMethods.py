import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


#Split the dataset
data = np.genfromtxt('data_breast_cancer.csv', delimiter=',')
x = data[:, :-1] # for all but last column
y = data[:, -1] # for last column

#Number of folds used in the Kfold-method throughout the script
FOLDS = 10

class DecisionTree:

    def __init__(self, max_depth: int, max_leaves: int) -> None:
        """
        Initialize the Decision Tree.
        :param max_depth: Maximum depth of the Decision Tree.
        :param max_leaves: Maximum number of leaf nodes in the Decision Tree
        :return: None
        """
        self.kf = createKFold()
        self.clf_dec = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=max_depth, max_leaf_nodes=max_leaves)

    def getAccuracy(self) -> float:
        """
        Get accuracy of the Decision Tree, measured in F1-score
        :return: The average F1-score over the given number of folds
        """
        cum_score = 0

        for train_index, test_index in self.kf.split(x):
            self.clf_dec.fit(x[train_index], y[train_index])

            cum_score = cum_score + f1_score(y[test_index], self.clf_dec.predict(x[test_index]))

        return 'Average f1 score over ', FOLDS, ' folds is: ', cum_score/FOLDS

    def printLearningCurve(self, sepIndex):
        """
        :param sepIndex: The number of datapoints used in the creation of the learning curve
        :print: The learning curve for the Deicision Tree
        """
        x_test = x[sepIndex:]
        y_test = y[sepIndex:]
        x_training = x[:sepIndex]
        y_training = y[:sepIndex]

        trainingdata_list = []
        score_list = []

        for i in range(5, len(x_training) + 1):
            self.clf_dec.fit(x_training[:i], y_training[:i])
            trainingdata_list.append(i)
            score_list.append(f1_score(y_test, self.clf_dec.predict(x_test)))

        plt.plot(trainingdata_list, score_list)
        plt.yticks(np.arange(0, 1, 0.05))
        plt.title("Decision Tree")
        plt.xlabel("Number of training data")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.show()



class NaiveBayes:

    def __init__(self) -> None:
        """
        Initialize the Gaussian Naive Bayes classifier.
        :return: None
        """

        self.kf = createKFold()
        self.clf_bayes = GaussianNB()

    def getAccuracy(self) -> float:
        cum_score = 0

        for train_index, test_index in self.kf.split(x):
            self.clf_bayes.fit(x[train_index], y[train_index])

            cum_score = cum_score + f1_score(y[test_index], self.clf_bayes.predict(x[test_index]))

        return 'Average f1 score over ', FOLDS, ' folds is: ', cum_score / FOLDS

    def printLearningCurve(self, sepIndex):
        """
        :param sepIndex: The number of datapoints used in the creation of the learning curve
        :print: The learning curve for the Deicision Tree
        """

        x_test = x[sepIndex:]
        y_test = y[sepIndex:]
        x_training = x[:sepIndex]
        y_training = y[:sepIndex]

        trainingdata_list = []
        score_list = []

        for i in range(5, len(x_training) + 1):
            self.clf_bayes.fit(x_training[:i], y_training[:i])
            trainingdata_list.append(i)
            score_list.append(f1_score(y_test, self.clf_bayes.predict(x_test)))

        plt.plot(trainingdata_list, score_list)
        plt.yticks(np.arange(0, 1, 0.05))
        plt.title("Naive Bayes")
        plt.xlabel("Number of training data")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.show()




def createKFold():
    """
    Creates an instance of Kfold from sklearn.model_selection. The data is shuffled before splitted into branches.
    :return: Instance of KFold
    """
    return KFold(n_splits=FOLDS, shuffle=True)

def decOptimalDepth(value):
    """
    Calculates the optimal depth value for a Decision Tree on the given input data
    :param value: Testing for maximum tree depths in range (2,value)
    :return: plots a curve comparing train and test data
    """
    kf = createKFold()
    values = [i for i in range(2,value+1)]
    train_scores = [];
    test_scores = [];

    for i in values:
        clf_dec = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_depth=i)

        cum_score_train = 0;
        cum_score_test = 0;
        for train_index, test_index in kf.split(x):
            clf_dec.fit(x[train_index], y[train_index])

            cum_score_test = cum_score_test + f1_score(y[test_index], clf_dec.predict(x[test_index]))
            cum_score_train = cum_score_train + f1_score(y[train_index], clf_dec.predict(x[train_index]))

        train_scores.append(cum_score_train/FOLDS)
        test_scores.append(cum_score_test/FOLDS)

    plotCurve(values, train_scores, test_scores, value, "Decision Tree", "Maximum depth", "F1-score")

def decOptimalLeaves(value):
    """
    Calculates the optimal number of leaf nodes for a Decision Tree on the given input data
    :param value: Testing for a number of maximum leaf nodes in range (2,value)
    :return: plots a curve comparing train and test data
    """
    kf = createKFold()
    values = [i for i in range(2,value+1)]
    train_scores = [];
    test_scores = [];

    for i in values:
        clf_dec = DecisionTreeClassifier(class_weight='balanced', criterion='entropy', max_leaf_nodes=i)

        cum_score_train = 0;
        cum_score_test = 0;
        for train_index, test_index in kf.split(x):
            clf_dec.fit(x[train_index], y[train_index])

            cum_score_test = cum_score_test + f1_score(y[test_index], clf_dec.predict(x[test_index]))
            cum_score_train = cum_score_train + f1_score(y[train_index], clf_dec.predict(x[train_index]))

        train_scores.append(cum_score_train/FOLDS)
        test_scores.append(cum_score_test/FOLDS)

    plotCurve(values, train_scores, test_scores, value, "Decision Tree", "Maximum number of leaf nodes", "F1-score")


def plotCurve(values_list, train_scores, test_scores, value, title: str, xlabel: str, ylabel: str):
    """
    Helper method used to plot the graphs in decOptimalDepth() and decOptimalLeaves()
    """
    plt.plot(values_list, train_scores, '-o', label='Train data')
    plt.plot(values_list, test_scores, '-o', label='Test data')
    plt.yticks(np.arange(0.85, 1, 0.05))
    plt.xticks(np.arange(0, value + 2, 2))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()



