import numpy as np
import csv

from sklearn.pipeline import Pipeline


def save_gridsearch_results(gs, filename):
    # if isinstance(gs, Pipeline):
    #     with open(filename, "w", newline='') as outfile:
    #         # first lines for model info
    #         outfile.write("best_model: {}\n".format(gs.named_steps))
    #         outfile.write("train_score: {:.2f} ({:.2f})\n".format(
    #             abs(gs["mean_train_score"]),
    #             abs(gs["std_train_score"])
    #         ))
    #         outfile.write("test_score: {:.2f} ({:.2f})\n".format(
    #             abs(gs["mean_test_score"]),
    #             abs(gs["std_test_score"])
    #         ))
    #         outfile.write("fit_time: {:.2f} ({:.2f})\n".format(
    #             abs(gs["mean_fit_time"]),
    #             abs(gs["std_fit_time"])
    #         ))
    # else:
        with open(filename, "w", newline='') as outfile:
            # first lines for model info
            outfile.write("best_model: {}\n".format(gs.best_estimator_.named_steps))
            outfile.write("train_score: {:.2f} ({:.2f})\n".format(
                abs(gs.cv_results_["mean_train_score"][gs.best_index_]),
                abs(gs.cv_results_["std_train_score"][gs.best_index_])
            ))
            outfile.write("test_score: {:.2f} ({:.2f})\n".format(
                abs(gs.cv_results_["mean_test_score"][gs.best_index_]),
                abs(gs.cv_results_["std_test_score"][gs.best_index_])
            ))
            outfile.write("fit_time: {:.2f} ({:.2f})\n".format(
                abs(gs.cv_results_["mean_fit_time"][gs.best_index_]),
                abs(gs.cv_results_["std_fit_time"][gs.best_index_])
            ))

            csvwriter = csv.writer(outfile, delimiter=',')
            # Create the header using the parameter names
            labels = ["mean_train_score", "std_train_score", 'mean_test_score', 'std_test_score',
                      'mean_fit_time', 'std_fit_time']

            param_names = list(gs.best_params_.keys())
            header = labels + param_names
            csvwriter.writerow(header)
            results = gs.cv_results_
            num_trials = len(results['params'])
            for i in range(num_trials):
                row = [results[label][i] for label in labels]
                for param in param_names:
                    row.append(results['param_' + param][i])
                csvwriter.writerow(row)
