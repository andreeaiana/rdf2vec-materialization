import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class DocumentSimilarityModel:

    def __init__(self, with_weights):
        self.with_weights = with_weights
        print("Document similarity model intialized.")

    def train(self, data, stats):
        doc_similarity = self._compute_doc_distance(data)
        gold_similarity_score, similarity_score = self._get_scores(
                stats, doc_similarity)
        pearson_score, spearman_score, harmonic_mean = self._evaluate_document_similarity(
                gold_similarity_score, similarity_score)
        if self.with_weights:
            conf = "with_weights"
        else:
            conf = "without_weights"
        return {"task_name": "DocumentSimilarity", "conf": conf,
                "pearson_score": round(pearson_score, 15),
                "spearman_score": round(spearman_score, 15),
                "harmonic_mean": round(harmonic_mean, 15)}

    def _compute_doc_distance(self, data):
        result_dict = {}
        doc1_list = list()
        doc2_list = list()
        doc_similarity = list()

        # Extract entities of document i and j
        for i in range(1, 51):
            set1 = self._extract_entities(data, i)
            weight1 = set1["weight"]
            if len(set1) == 0:
                print("Document Similarity: No entities in doc " + str(i) +
                      "\n")
                continue

            for j in range(1, 51):
                set2 = self._extract_entities(data, j)
                weight2 = set2["weight"]
                if len(set2) == 0:
                    print("Document Similarity: No entities in doc " + str(j)
                          + "\n")
                    continue

            # Compute the similarity for each pair of entities in d_i and d_j
            distance_score1 = self._compute_distance(set1, set2)
            distance_score2 = self._compute_distance(set2, set1)

            # For each entity in d_i, identify the maximum similarity to an
            # entity in d_j, and viceversa
            max_sim1 = list()
            for k in range(len(distance_score1)):
                weight = weight1.iloc[k]
                max_sim1.append(self._compute_max_similarity(
                        distance_score1[k, :], weight, weight2))

            max_sim2 = list()
            for k in range(len(distance_score2)):
                weight = weight2.iloc[k]
                max_sim2.append(self._compute_max_similarity(
                        distance_score2[k, :], weight, weight1))

            # Calculate document similarity
            sum_max_sim1 = sum(max_sim1)
            sum_max_sim2 = sum(max_sim2)
            document_similarity = (sum_max_sim1 + sum_max_sim2)/(
                    len(set1) + len(set2))

            doc1_list.append(i)
            doc2_list.append(j)
            doc_similarity.append(document_similarity)

        result_dict["doc1"] = doc1_list
        result_dict["doc2"] = doc2_list
        result_dict["similarity"] = doc_similarity

        return pd.DataFrame(result_dict)

    def _get_scores(self, gold_stats, actual_stats):
        merged = pd.merge(gold_stats, actual_stats, on=["doc1", "doc2"],
                          how="inner")
        return (merged.iloc[:, 2], merged.iloc[:, 3])

    def _evaluate_document_similarity(self, gold_simialrity_score,
                                      similarity_score):
        pearson_score, _value = pearsonr(gold_simialrity_score,
                                         similarity_score)
        spearman_score, _value = spearmanr(gold_simialrity_score,
                                           similarity_score)
        harmonic_mean = (2*pearson_score*spearman_score)/(
                pearson_score + spearman_score)
        return pearson_score, spearman_score, harmonic_mean

    def _extract_entities(self, data, documentID):
        set1 = data[data["doc"] == documentID]
        if len(set1) == 0:
            print("No entities in doc " + str(documentID))
        else:
            set1 = set1.sort_values("weight", ascending=False).drop_duplicates(
                    subset="id", keep="first")
        return set1

    def _compute_distance(self, set1, set2):
        return pairwise_distances(set1.iloc[:, 3:], set2.iloc[:, 3:],
                                  metric="cosine")

    def _compute_max_similarity(self, distance_list, weight1, weight_list):
        index_min_distance = np.argmin(distance_list)
        min_distance_score = distance_list[index_min_distance]
        max_distance = max(distance_list)
        if max_distance != 0:
            normalized_distance = min_distance_score/max_distance
        else:
            normalized_distance = min_distance_score
        similarity_score = 1-normalized_distance

        if self.with_weights:
            weight2 = weight_list.iloc[index_min_distance]
            similarity_score = similarity_score * (weight1 * weight2)

        return similarity_score
