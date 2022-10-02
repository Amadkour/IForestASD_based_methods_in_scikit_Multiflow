from source import functions
import datetime

func = functions.Comparison()

window_sizes = [500]
n_estimators = [10, ]
anomaly_threshold = 0.5
max_sample = 10000  # We have gotten the size of the min dataset (Shuttle) to evaluate all dataset on the same basis.
n_wait = max_sample  # The evaluation step size
# Used metric in the evaluation. Attention to use the metrics availlable in skmultiflow
metrics = ['accuracy', 'f1', 'precision', 'recall', 'true_vs_predicted', 'kappa', 'kappa_m', 'running_time',
           'model_size']
# metrics=['accuracy']
dataset_name = "SMTP2"
test_name = dataset_name + '_' + str(datetime.datetime.now())
drift_rate = 0.03
stream = func.get_dataset(dataset_name=dataset_name)
for window in window_sizes:
    for n_estimator in n_estimators:
        print("")
        print("******************************** Window = " + str(window) + " and n_estimator = " + str(
            n_estimator) + " ********************************")
        func.run_comparison(stream=stream, window=window,
                            estimators=n_estimator, anomaly=anomaly_threshold, drift_rate=drift_rate,
                            result_folder=test_name, max_sample=max_sample, n_wait=n_wait, metrics=metrics)
