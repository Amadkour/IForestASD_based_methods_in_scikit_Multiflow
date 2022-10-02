from source.iforestasd_adwin_scikitmultiflow import path_length_tree, IsolationTree
from skmultiflow.utils import check_random_state

import numpy as np

import random
import threading
from skmultiflow.drift_detection.adwin import ADWIN


class IsolationForestStreamImprove:

    def __init__(self, window_size=100, n_estimators=25, anomaly_threshold=0.5,
                 drift_threshold=0.5, random_state=None, version="PADWIN",
                 # Parameters for partial model update
                 n_estimators_updated=0.5, updated_randomly=True):

        super().__init__()
        self.model_update = []
        self.is_learning_phase_on = None
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.window_size = window_size

        self.samples_seen = 0

        self.anomaly_rate = 0.20

        self.anomaly_threshold = anomaly_threshold

        self.drift_threshold = drift_threshold
        self.window = None

        self.prec_window = None

        self.cpt = 0
        self.version = version
        # To count the number of times the model have been updated 0 Not updated and 1 updated
        self.model_update_windows = []
        # To count the number of times the model have been updated 0 Not updated and
        # 1 updated
        self.model_update.append(version)  # Initialisation to know the concerned version of IForestASD
        self.model_update_windows.append(
            "samples_seen_" + version)  # Initialisation to know the number of data seen in the window
        self.n_estimators_updated = int(
            self.n_estimators * n_estimators_updated)  # The percentage of new trees to compute when update on new
        # window
        if n_estimators_updated <= 0.0 or n_estimators_updated > 1.0:
            raise ValueError("n_estimators_updated must be > 0 and <= 1")

        self.updated_randomly = updated_randomly  # If we choose randomly the trees: True for randomly,
        # False to pick the first (n_estimators- int(n_estimators*n_estimators_updated)) trees

        self.first_time_fit = True
        self.adwin = ADWIN()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        global ensemble
        number_instances, _ = X.shape

        if self.samples_seen == 0:
            iforest = IsolationTreeEnsemble(self.window_size, self.n_estimators, self.random_state)
            ensemble = iforest
        for i in range(number_instances):
            self._partial_fit(X[i], y)

        return self

    def _partial_fit(self, X, y):
        global ensemble
        X = np.reshape(X, (1, len(X)))

        if self.samples_seen % self.window_size == 0:
            # Update the two windows (precedent one and current windows)
            self.prec_window = self.window
            self.window = X
        else:
            self.window = np.concatenate((self.window, X))

        if self.samples_seen % self.window_size == 0 and self.samples_seen != 0:
            if self.first_time_fit:  # It is the first window
                ensemble.fit(self.prec_window)
                self.first_time_fit = False
            if self.version == "PADWIN":
                prec_window_predictions = self.predict_simple(self.prec_window)
                drift_detected = False
                for pred in prec_window_predictions:
                    self.adwin.add_element(pred)
                    if self.adwin.detected_change():
                        drift_detected = True
                        break
                if drift_detected:
                    print("===============[ threads ]===================")
                    # online(self.adwin, prec_window_predictions, prec_window= self.prec_window)
                    # t1 = threading.Thread(target=online, args=(self.adwin, prec_window_predictions,))
                    t2 = threading.Thread(target=offline,
                                          args=(self.adwin, self.prec_window, prec_window_predictions,
                                                self.window_size, self.n_estimators,
                                                self.random_state))
                    # t1.start()
                    t2.start()

                    self.model_update.append(str(1))
                    self.model_update_windows.append(self.samples_seen.__str__())
                    self.adwin.reset()
                else:
                    self.model_update.append(str(0))
                    self.model_update_windows.append(self.samples_seen.__str__())
        self.samples_seen += 1

    def partial_update_model(self, window, y):
        global ensemble
        self.is_learning_phase_on = True
        iforest = IsolationTreeEnsemble(self.window_size, self.n_estimators_updated, self.random_state)
        iforest.fit(window)
        if self.updated_randomly:
            old_trees_idx = random.sample(list(range(len(ensemble.trees))),
                                          (self.n_estimators - self.n_estimators_updated))
        else:
            old_trees_idx = range(self.n_estimators - self.n_estimators_updated)

        for t in old_trees_idx:
            iforest.trees.append(ensemble.trees[t])
        ensemble.trees = iforest.trees

    def update_model(self, window):
        global ensemble
        self.is_learning_phase_on = True
        iforest = IsolationTreeEnsemble(self.window_size, self.n_estimators, self.random_state)
        ensemble = iforest
        ensemble.fit(window)

    def anomaly_scores_rate(self, window):
        global ensemble
        score_tab = 2.0 ** (-1.0 * ensemble.path_length(window) / c(len(window)))
        score = 0
        for x in score_tab:
            if x > self.anomaly_threshold:
                score += 1
        return score / len(score_tab)

    def predict_simple(self, X):
        global ensemble
        # print(self.anomaly_threshold)
        prediction = ensemble.predict_from_instances_scores(ensemble.anomaly_score(X), self.anomaly_threshold)
        # return prediction of all instances
        return prediction

    def predict(self, X):
        global ensemble
        if self.samples_seen <= self.window_size:
            return [-1]  # Return the last element

        X = np.reshape(X, (1, len(X[0])))
        self.prec_window = np.concatenate((self.prec_window, X))  # Append the instances in the sliding window
        a = self.predict_proba()
        prediction = ensemble.predict_from_anomaly_scores(a,
                                                          self.anomaly_threshold)
        return [prediction]

    def predict_proba(self):
        global ensemble
        if self.samples_seen <= self.window_size:
            return [-1]
        return ensemble.anomaly_score(self.prec_window)[
            -1]  # Anomaly return an array with all scores of each data, taking -1 return the last instance (X)
        # anomaly score


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees, random_state):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.depth = np.log2(sample_size)
        self.trees = []
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.is_learning_phase_on = True

    def fit(self, X: np.ndarray):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        len_x = len(X)

        for i in range(self.n_trees):
            sample_idx = random.sample(list(range(len_x)), self.sample_size)
            temp_tree = IsolationTree(self.depth, 0).fit(X[sample_idx])
            self.trees.append(temp_tree)

        return self

    def path_length(self, X: np.ndarray):
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []

        for x in X:
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)
        return pl_vector

    def anomaly_score(self, X: np.ndarray):

        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores: int, threshold: float):
        predictions = 1 if scores >= threshold else 0

        return predictions

    def predict_from_instances_scores(self, scores: np.ndarray, threshold: float) -> list[int]:
        predictions = [1 if p[0] >= threshold else 0 for p in scores]
        return predictions


class MyPool:
    def __init__(self):
        self.current_model = None
        self.future_model = None
        self.old_models = []
        self.environment = []

    def get_similar(self, data):
        similarity = self.euclidean_distance(data, self.environment[0])
        similar = self.old_models[0]
        for index in range(len(self.environment)):

            s = self.euclidean_distance(data, self.environment[index])
            if similarity > s:
                similarity = s
                similar = self.old_models[index]
        return similar

    def euclidean_distance(self, x, y):
        from math import pow, sqrt
        return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

    def get_statistics(self, data):
        data = np.array(data)
        total_sd = 0
        total_min = 0
        total_max = 0
        total_mean = 0
        total_h_mean = 0
        length = len(data[0])
        import statistics
        for i in range(length):
            d = data[:, i]
            total_sd += statistics.stdev(d)
            total_mean += statistics.mean(d)
            total_h_mean += statistics.harmonic_mean(d)
            total_max += max(d)
            total_min += min(d)

        return [total_sd / float(length), total_mean / float(length), total_h_mean / float(length),
                total_min / float(length), total_max / float(length)]


def predict_future_environment(data):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    data = np.array(data[:][0:1]).reshape(-1, 1).tolist()
    print(data)
    data = data
    print(data)
    model = SARIMAX(data, order=(1, 1,1))
    model_fit = model.fit()

    # make prediction
    return model_fit.forecast()


def online(adwin, prec_window_predictions,prec_window):
    print('---------online')

    global ensemble
    global my_pool
    if len(my_pool.old_models) > 2:
        print('---------online')
        e = ensemble
        ensemble = my_pool.get_similar(my_pool.get_statistics(prec_window))
        drift_detected = False
        for pred in random.sample(prec_window_predictions, 100):
            adwin.add_element(pred)
            if adwin.detected_change():
                drift_detected = True
                break
        if drift_detected:
            ensemble = e


def offline(adwin, prec_window, prec_window_predictions, window_size, sn_estimators,
            random_state):
    global ensemble
    global my_pool
    if len(my_pool.old_models) < 2:
        print('---------update')
        update_model2(prec_window, window_size, sn_estimators, random_state)
    else:
        ensemble = my_pool.get_similar(my_pool.get_statistics(prec_window))
        print('---------loaded')
        drift_detected = False
        for pred in random.sample(prec_window_predictions, 100):
            adwin.add_element(pred)
            if adwin.detected_change():
                drift_detected = True
                break
        if drift_detected is True:
            update_model2(prec_window, window_size, sn_estimators, random_state)
    # my_pool.future_model = my_pool.get_similar(predict_future_environment(my_pool.environment))
    # my_pool.future_model = my_pool.get_similar(prec_window)
    my_pool.old_models.append(ensemble)
    my_pool.environment.append(my_pool.get_statistics(prec_window))


ensemble: IsolationTreeEnsemble
my_pool = MyPool()


def update_model2(window, window_size, sn_estimators, random_state):
    global ensemble
    iforest = IsolationTreeEnsemble(window_size, sn_estimators, random_state)
    ensemble = iforest
    ensemble.fit(window)


def c(n):
    if n > 2:
        return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1.) / (n * 1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0