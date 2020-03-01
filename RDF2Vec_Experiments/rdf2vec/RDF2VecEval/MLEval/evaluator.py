from model import ClassificationModel, RegressionModel
import os
import csv
import pandas as pd
from collections import defaultdict
import codecs
from data_manager import data_manager
from numpy import mean

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class Evaluator:

    def __init__(self, w2v_model_name, task_name):
        self.w2v_model_name = w2v_model_name
        self.task_name = task_name

        self.results_folder = "../../../../Data/processed/" \
            + "evaluation_results/" + self.w2v_model_name
        if not os.path.isdir(self.results_folder):
            os.mkdir(self.results_folder)

        self.vectors_folder = "../../../../Data/processed/" \
            + "evaluation_vectors/" + self.w2v_model_name
        if not os.path.isdir(self.vectors_folder):
            os.mkdir(self.vectors_folder)

    def evaluate(self):
        log_errors = ""
        gold_standard_filenames = ["CitiesQualityOfLiving", "AAUP",
                                   "Forbes2013", "MetacriticMovies",
                                   "MetacriticAlbums"]
        total_scores = defaultdict(dict)
        if self.task_name == "Classification":
            column_score_name = "label"
            model_names = ["NB", "KNN", "C45", "SVM"]
            SVM_configurations = [pow(10, -3), pow(10, -2), 0.1, 1.0, 10.0,
                                  pow(10, 2), pow(10, 3)]
            Model = ClassificationModel
        elif self.task_name == "Regression":
            column_score_name = "rating"
            model_names = ["LR", "KNN", "M5"]
            Model = RegressionModel
        else:
            print("WRONG TASK NAME!")

        for gold_standard_filename in gold_standard_filenames:
            print("Evaluating dataset: {}.\n\n".format(gold_standard_filename))
            gold_standard_file = "../../../../Data/raw/evaluation_sets/" + \
                gold_standard_filename + "/CompleteDataset15.tsv"
            vectors_file = "../../../../Data/processed/evaluation_vectors/" \
                + self.w2v_model_name + "/" + gold_standard_filename + ".txt"
            data_mng = data_manager(gold_standard_file, vectors_file,
                                    self.w2v_model_name)
            vectors = data_mng.retrieve_vectors()
            scores = defaultdict(list)
            total_scores_element = defaultdict(list)
            data, ignored = data_mng.intersect_vectors_goldStandard(
                    vectors, column_score_name)
            self._store_ignored(gold_standard_filename, ignored)

            if data.size == 0:
                log_errors += "Classification : Problems in merging vector "
                "with gold standard " + gold_standard_file + "\n"
                print("Classification : Problems in merging vector with gold "
                      "standard " + gold_standard_file + "\n")
            else:
                for i in range(10):
                    data = data.sample(frac=1, random_state=i).reset_index(
                            drop=True)
                    for model_name in model_names:
                        if model_name != "SVM":
                            # Initialize the model
                            model = Model(model_name)
                            # Train and print score
                            try:
                                result = model.train(data)
                                result["gold_standard_file"] = \
                                    gold_standard_filename
                                scores[model_name].append(result)
                                total_scores_element[model_name].append(result)
                            except Exception as e:
                                log_errors += "File used as gold standard" + \
                                    gold_standard_filename + "\n"
                                log_errors += str(self.task_name) + \
                                    " method: " + model_name + "\n"
                                log_errors += str(e) + "\n"
                        else:
                            for conf in SVM_configurations:
                                # Initialize the model
                                model = Model("SVM", conf)
                                try:
                                    result = model.train(data)
                                    result['gold_standard_file'] = \
                                        gold_standard_filename
                                    scores["SVM"].append(result)
                                    total_scores_element["SVM_" + str(conf)
                                                         ].append(result)
                                except Exception as e:
                                    log_errors += "File used as gold " \
                                        + "standard: " \
                                        + gold_standard_filename + "\n"
                                    log_errors += str(self.task_name) \
                                        + " method: SVM " + str(conf) + "\n"
                                    log_errors += str(e) + "\n"
                self._store_results(gold_standard_filename, scores)
                total_scores[gold_standard_filename] = total_scores_element

        results_df = self._resultsAsDataFrame(total_scores)
        return results_df

    def _store_ignored(self, gold_standard_filename, ignored):
        file_ignored = codecs.open(self.vectors_folder + "/"
                                   + str(self.task_name) + "_"
                                   + gold_standard_filename
                                   + "_ignoredData.txt", "w", "latin1")
        file_ignored.write("Classification : Ignored data: " +
                           str(len(ignored)) + "\n")
        for ignored_tuple in ignored.itertuples():
            value = ignored_tuple[2]
            file_ignored.write(value + '\n')
        file_ignored.close()

    def _store_results(self, gold_standard_filename, scores):
        with open(self.results_folder + "/" + str(self.task_name) + "_" +
                  gold_standard_filename + "_results.csv", "w") as csv_file:
            fieldnames = ["task_name", "gold_standard_file", "model_name",
                          "model_configuration", str(self._get_metric_list()[0]
                          )]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for (method, scoresForMethod) in scores.items():
                for score in scoresForMethod:
                    writer.writerow(score)

    def _resultsAsDataFrame(self, scores):
        data_dict = dict()
        data_dict["task_name"] = list()
        data_dict["gold_standard_file"] = list()
        data_dict["model_name"] = list()
        data_dict["model_configuration"] = list()
        data_dict["metric"] = list()
        data_dict["score_value"] = list()

        metrics = self._get_metric_list()

        for (gold_standard_filename, gold_standard_scores) in scores.items():
            for (method, scoresForMethod) in gold_standard_scores.items():
                for metric in metrics:
                    metric_scores = list()
                    for score in scoresForMethod:
                        metric_scores.append(score[metric])
                    metric_score = mean(metric_scores)

                    score = scoresForMethod[0]
                    configuration = score["model_configuration"]
                    if configuration is None:
                        configuration = "-"

                    data_dict["task_name"].append(score["task_name"])
                    data_dict["gold_standard_file"].append(score[
                            "gold_standard_file"])
                    data_dict["model_name"].append(score["model_name"])
                    data_dict["model_configuration"].append(configuration)
                    data_dict["metric"].append(metric)
                    data_dict["score_value"].append(metric_score)

        results_df = pd.DataFrame(data_dict, columns=[
                "task_name", "gold_standard_file", "model_name",
                "model_configuration", "metric", "score_value"])
        results_file = self.results_folder + "/" + str(self.task_name) \
                        + "_average_results.csv"
        results_df.to_csv(results_file, index=False)
        return results_df

    def _get_metric_list(self):
        if self.task_name == "Classification":
            return ["accuracy"]
        elif self.task_name == "Regression":
            return ["root_mean_squared_error"]
        else:
            print("WRONG TASK NAME!")

    def main():
        import argparse
        parser = argparse.ArgumentParser(
                description='Arguments for ML evaluation.')
        parser.add_argument('w2v_model_name',
                            choices=["DB2Vec_500w_4d_500v",
                                     "DB2Vec_500w_4d_200v",
                                     "DB2Vec_500w_4d_500v_enriched",
                                     "DB2Vec_500w_4d_200v_enriched",
                                     "DB2Vec_500w_4d_500v_enriched_dllearner",
                                     "DB2Vec_500w_4d_200v_enriched_dllearner",
                                     "DB2Vec_500w_8d_500v",
                                     "DB2Vec_500w_8d_200v",
                                     "DB2Vec_500w_8d_500v_enriched",
                                     "DB2Vec_500w_8d_200v_enriched",
                                     "DB2Vec_500w_8d_500v_enriched_dllearner",
                                     "DB2Vec_500w_8d_200v_enriched_dllearner",
                                     "WD2Vec_200w_4d_500v",
                                     "WD2Vec_200w_4d_200v",
                                     "WD2Vec_200w_4d_500v_enriched",
                                     "WD2Vec_200w_4d_200v_enriched"],
                            help='the word2vec model used')
        parser.add_argument("task_name",
                            choices=["Classification", "Regression"],
                            help="ML task")
        args = parser.parse_args()
        from evaluator import Evaluator
        evaluator = Evaluator(args.w2v_model_name, args.task_name)
        evaluator.evaluate()

    if __name__ == "__main__":
        main()
