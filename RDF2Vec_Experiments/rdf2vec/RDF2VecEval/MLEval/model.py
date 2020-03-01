from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class ClassificationModel:
    """
    It initializes the model for the classification task.

    model_name: name of the classification model to train
    C_value: the value of the C variable for the SVM model (read-only if the
                                                            model is SVM)
    """
    def __init__(self, model_name, C_value=None):
        self.model_name = model_name
        self.configuration = None

        # Create the model
        if self.model_name == "NB":
            self.model = GaussianNB()
        elif self.model_name == "KNN":
                self.model = KNeighborsClassifier(n_neighbors=3)
                self.configuration = "K=3"
        elif self.model_name == "SVM":
            if C_value is None:
                raise Exception("For SVM, the C value has to be " +
                                "specified. The default value is 1.0.")
            self.model = SVC(C=C_value)
            self.configuration = "C=" + str(C_value)
        elif self.model_name == "C45":
            self.model = DecisionTreeClassifier()
        else:
            print("YOU CHOSE WRONG MODEL FOR CLASSIFICATION!")
        print("Classification model initialized.")

    """
    It trains the classification model for the provided dataset.

    Parameters:
        data:
            dataframe with entity name as first column, class label as second
            column and the vectors starting from the third column

    Return:
        The model name, its configuration (if any), and the model's accuracy.
    """
    def train(self, data):
        print("Training {} model.".format(self.model_name))
        scoring = "accuracy"
        n_splits = 10
        n_samples = data.shape[0]
        if n_splits > n_samples:
            raise ValueError(
                ("Classification : Cannot have number of splits n_splits={} "
                 "greater than the number of samples: {}.").format(
                         n_splits, n_samples) + "\n")
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["label"],
                                 cv=n_splits, scoring=scoring)
        print("Model trained.\n")
        scoring_value = np.mean(scores)
        print("Classification - model_name: {}, configuration: {} "
              "scoring: {}, score: {}".format(self.model_name,
                        self.configuration, scoring, round(scoring_value, 15)))
        return {"task_name": "Classification", "model_name": self.model_name,
                "model_configuration": self.configuration,
                scoring: round(scoring_value, 15)}


class RegressionModel:
    """
    It initializes the model for the regression task.

    model_name: name of the regression model to train
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.configuration = None

        # Create the model
        if self.model_name == "LR":
            self.model = LinearRegression()
        elif self.model_name == "M5":
            self.model = DecisionTreeRegressor()
        elif self.model_name == "KNN":
            self.model = KNeighborsRegressor(n_neighbors=3)
            self.configuration = "K=3"
        else:
            print("YOU CHOSE WRONG MODEL FOR REGRESSION!")

    def train(self, data):
        """
        It trains the regression model for the provided dataset.

        Parameters:
            data:
                dataframe with entity name as first column, class label as
                second column and the vectors starting from the third column

        Return:
            The model name, its configuration (if any), and the model's
            accuracy.
        """
        print("Training {} model.".format(self.model_name))
        scoring = "neg_mean_squared_error"
        n_splits = 10
        n_samples = data.shape[0]
        if n_splits > n_samples:
            raise ValueError(
                ("Regression : Cannot have number of splits n_splits={} "
                 "greater than the number of samples: {}.").format(
                         n_splits, n_samples) + "\n")
        scores = cross_val_score(self.model, data.iloc[:, 2:], data["rating"],
                                 cv=n_splits, scoring=scoring)
        scoring = "root_mean_squared_error"
        scoring_value = np.mean(np.sqrt(np.abs(scores)))
        print("Regression - model_name: {}, configuration: {} "
              "scoring: {}, score: {}".format(self.model_name,
                        self.configuration, scoring, round(scoring_value, 15)))
        return {"task_name": "Regression", "model_name": self.model_name,
                "model_configuration": self.configuration,
                scoring: round(scoring_value, 15)}
