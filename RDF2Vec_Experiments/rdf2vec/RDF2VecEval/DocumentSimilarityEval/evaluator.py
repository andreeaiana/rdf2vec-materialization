import os
import csv
import codecs
import pandas as pd
from collections import defaultdict
from data_manager import data_manager
from model import DocumentSimilarityModel

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class Evaluator:

    def __init__(self, w2v_model_name):
        self.w2v_model_name = w2v_model_name

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
        vectors_file = "../../../../Data/processed/evaluation_vectors/" \
            + self.w2v_model_name + "/LP50.txt"
        stats_file = "../../../../Data/raw/evaluation_sets/LP50/" \
            + "LP50_averageScores.csv"
        data_mng = data_manager(vectors_file, self.w2v_model_name)
        stats = data_mng.read_file(stats_file, ["doc1", "doc2", "average"])
        vectors = data_mng.retrieve_vectors()
        data, ignored = data_mng.intersect_vectors_goldStandard(vectors)
        self._store_ignored(ignored)

        scores = defaultdict(dict)
        if data.size == 0:
            log_errors += "Document similarity : Problems in merging vector " \
                        + "with gold standard LP50.\n"
            print("Document similarity : Problems in merging vector with gold"
                  + " standard LP50.\n")
        else:
            try:
                with_weights = False
                model = DocumentSimilarityModel(with_weights)
                result = model.train(data, stats)
                scores["without_weights"] = result

                with_weights = True
                model = DocumentSimilarityModel(with_weights)
                result = model.train(data, stats)
                scores["with_weights"] = result
                self._store_results(scores)
                results_df = self._resultAsDataFrame(scores)
            except Exception as e:
                log_errors += "File used as gold standard: LP50.\n"
                log_errors += "Document similarity, with weights: " \
                    + str(with_weights) + "\n"
                log_errors += str(e) + "\n"

    def _store_ignored(self, ignored):
        ignored = ignored.drop_duplicates()
        results_file = self.vectors_folder \
            + "/DocumentSimilarity_LP50_ignoredData.txt"
        file_ignored = codecs.open(results_file, "wb", "latin1")
        file_ignored.write("Document similarity: Ignored data : "
                           + str(len(ignored)) + "\n\n")
        for ignored_tuple in ignored.itertuples():
            value = ignored_tuple[2]
            file_ignored.write(value + '\n')
        file_ignored.close()

    def _store_results(self, scores):
        with open(self.results_folder + "/DocumentSimilarity_LP50_results.csv",
                  "w", newline="") as csv_file:
            fieldnames = ["task_name", "conf", "pearson_score",
                          "spearman_score", "harmonic_mean"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for (method, score) in scores.items():
                writer.writerow(score)

    def _resultAsDataFrame(self, scores):
        data_dict = dict()
        data_dict["task_name"] = list()
        data_dict["model"] = list()
        data_dict["metric"] = list()
        data_dict["score_value"] = list()

        metrics = ['pearson_score', 'spearman_score', 'harmonic_mean']

        for (configuration, score) in scores.items():
            for metric in metrics:
                data_dict["task_name"].append(score["task_name"])
                data_dict["model"].append(score["conf"])
                data_dict["metric"].append(metric)
                data_dict["score_value"].append(score[metric])

        results_df = pd.DataFrame(data_dict, columns=["task_name", "model",
                                                      "metric", "score_value"])
        results_file = self.results_folder + "/DocumentSimilarity_results.csv"
        results_df.to_csv(results_file, index=False)
        return results_df

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
        args = parser.parse_args()
        from evaluator import Evaluator
        evaluator = Evaluator(args.w2v_model_name)
        evaluator.evaluate()

    if __name__ == "__main__":
        main()
