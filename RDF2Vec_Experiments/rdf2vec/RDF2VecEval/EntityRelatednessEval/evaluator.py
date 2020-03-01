import os
import pandas as pd
from numpy import mean
import unicodecsv as csv
from data_manager import data_manager
from model import EntityRelatednessModel

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
        vectors_file_input = "../../../../Data/processed/evaluation_vectors/" \
                + self.w2v_model_name + "/KORE_input_vectors.txt"
        vectors_file_output = "../../../../Data/processed/evaluation_vectors/" \
                + self.w2v_model_name + "/KORE_output_vectors.txt"
        data_mng = data_manager(vectors_file_input, vectors_file_output,
                                self.w2v_model_name)
        input_vectors, output_vectors = data_mng.retrieve_vectors()
        groups = data_mng.read_file()
        scores = list()
        left_entities_df = pd.DataFrame({"id": list(groups.keys())})
        left_merged, left_ignored = data_mng.intersect_vectors_goldStandard(
                output_vectors, left_entities_df)

        self._store_ignored(left_ignored)

        if left_merged.size == 0:
            log_errors += "EntityRelatedness: no left entities of KORE in " \
                        + "vectors.\n"
            print("EntityRelatedness: no left entities of KORE in vectors.\n")
        else:
            right_merged_list = list()
            right_ignored_list = list()

            for key in groups.keys():
                right_entities_df = pd.DataFrame({"id": groups[key]})
                right_merged, right_ignored = \
                                data_mng.intersect_vectors_goldStandard(
                                        input_vectors, right_entities_df)
                right_ignored["related_to"] = key
                right_merged_list.append(right_merged)
                right_ignored_list.append(right_ignored)

                self._store_ignored(right_ignored)
            model = EntityRelatednessModel()
            scores = model.train(left_merged, left_ignored,
                                 right_merged_list, right_ignored_list, groups)

            self._store_results(scores)
            results_df = self._resultAsDataFrame(scores)

    def _store_ignored(self, ignored):
        results_file = self.vectors_folder \
                + "/EntityRelatedness_KORE_ignoredData.csv"
        with open(results_file, "ab") as csv_file:
            fieldnames = ["entity", "related_to"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not os.path.isfile(results_file):
                writer.writeheader()
            for ignored_tuple in ignored.itertuples():
                if "related_to" in ignored.columns:
                    related_to_value = ignored_tuple[-1]
                else:
                    related_to_value = ''
                value = ignored_tuple[1]
                writer.writerow({"entity": value, "related_to":
                                 related_to_value})

    def _store_results(self, scores):
        with open(self.results_folder + "/EntityRelatedness_KORE_results.csv",
                  "wb") as csv_file:
            fieldnames = ["task_name", "entity_name",
                          "spearmanr_correlation", "spearmanr_pvalue"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for score in scores:
                writer.writerow(score)

    def _resultAsDataFrame(self, scores):
        data_dict = dict()
        metric = "spearmanr_correlation"

        metric_scores = dict()
        metric_scores["All"] = list()
        metric_scores["IT Companies"] = list()
        metric_scores["Hollywood Celebrities"] = list()
        metric_scores["Television Series"] = list()
        metric_scores["Video Games"] = list()
        metric_scores["Chuck Norris"] = list()

        i = 0
        for score in scores:
            metric_scores["All"].append(score[metric])
            if i < 5:
                metric_scores["IT Companies"].append(score[metric])
            elif i < 10:
                metric_scores["Hollywood Celebrities"].append(score[metric])
            elif i < 15:
                metric_scores["Video Games"].append(score[metric])
            elif i == 15:
                metric_scores["Chuck Norris"].append(score[metric])
            else:
                metric_scores["Television Series"].append(score[metric])
            i += 1

        metric_score = dict()
        for key in metric_scores.keys():
            metric_score[key] = mean(metric_scores[key])

        score = scores[0]

        data_dict["task_name"] = score["task_name"]
        data_dict["metric"] = metric
        data_dict["All_score_value"] = metric_score["All"]
        data_dict["IT_Companies_score_value"] = metric_score["IT Companies"]
        data_dict["Hollywood_Celebrities_score_value"] = metric_score[
                "Hollywood Celebrities"]
        data_dict["Television_Series_score_value"] = metric_score[
                "Television Series"]
        data_dict["Video_Games_score_value"] = metric_score["Video Games"]
        data_dict["Chuck_Norris_score_value"] = metric_score["Chuck Norris"]
        results_df = pd.DataFrame(data_dict, columns=[
                "task_name", "metric", "All_score_value",
                "IT_Companies_score_value",
                "Hollywood_Celebrities_score_value",
                "Television_Series_score_value", "Video_Games_score_value",
                "Chuck_Norris_score_value"], index=[0])
        results_file = self.results_folder + "/EntityRelatedness_" \
                        + "average_results.csv"
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
