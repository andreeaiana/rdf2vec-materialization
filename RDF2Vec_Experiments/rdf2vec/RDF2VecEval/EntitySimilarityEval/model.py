import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class EntitySimilarityModel:
    def __init__(self):
        print("Entity similarity model initialized.")

    def train(self, left_merged, left_ignored, right_merged_list,
              right_ignored_list, groups):
        predicted_ranking_list = self._compute_similarity(
                left_merged, left_ignored, right_merged_list,
                right_ignored_list)
        gold_rank_list = np.tile(np.arange(1, 21), (21, 1))
        return self._evaluate_ranking(list(groups.keys()), gold_rank_list,
                                      predicted_ranking_list)

    def _compute_similarity(self, left_merged, left_ignored,
                            right_merged_list, right_ignored_list):
        predicted_rank_list = list()

        for i in range(len(left_merged["id"])):
            distances = []
            if right_merged_list[i].size > 0:
                distances = distance.cdist(left_merged.iloc[[i], 1:],
                                           right_merged_list[i].iloc[:, 1:],
                                           metric="cosine").flatten()
            ignored_distances = np.array([1 for j in range(len(
                    right_ignored_list[i]))])
            distances = np.concatenate((distances, ignored_distances))
            predicted_rank_list.append(list(np.argsort(distances)))

        for i in range(len(left_ignored["id"])):
            predicted_rank_list.append(list(np.argsort([1 for i in range(20)])
                                        ))

        return predicted_rank_list

    def _evaluate_ranking(self, entities_list, gold_ranking_list,
                          predicted_ranking_list):
        scores_list = list()
        for i in range(len(entities_list)):
#            print("Entity similarity : " +  entities_list[i])
#            print(gold_ranking_list[i])
#            print(predicted_ranking_list[i])
            spearmanr_correlation, spearmanr_pvalue = spearmanr(
                    gold_ranking_list[i], predicted_ranking_list[i])
#            print("Entity similarity : " + entities_list[i]
#                  + " kendalltau correlation " + str(spearmanr_correlation)
#                  + " kendall tau pvalue " + str(spearmanr_pvalue))
            scores_list.append({"task_name": "EntitySimilarity",
                                "entity_name": entities_list[i],
                                "spearmanr_correlation": round(
                                       spearmanr_correlation, 15),
                                "spearmanr_pvalue": round(spearmanr_pvalue, 15)
                                })

        return scores_list
