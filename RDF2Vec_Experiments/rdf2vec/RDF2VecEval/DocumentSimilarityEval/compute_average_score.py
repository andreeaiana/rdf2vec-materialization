from collections import defaultdict
import csv

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


if __name__ == "__main__":
    combined = defaultdict(list)
    stats_file = "../../../../Data/raw/evaluation_sets/LP50/LP50_stats.csv"
    with open(stats_file) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            else:
                # SubjectID,Document1,Document2,Similarity,Time
                parts = line.split(",")
                doc1 = int(parts[1])
                doc2 = int(parts[2])
                score = int(parts[3])
                if doc1 > doc2:
                    doc1, doc2 = doc2, doc1
                combined[(doc1, doc2)].append(score)

    average_scores_file = "../../../../Data/raw/evaluation_sets/LP50/" \
                          + "LP50_averageScores.csv"
    with open(average_scores_file, "w", newline="") as csv_file:
        fieldnames = ["doc1", "doc2", "average"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for doc1 in range(1, 51):
            for doc2 in range(doc1 + 1, 51):
                score = combined[(doc1, doc2)]
                if len(score) == 0:
                    raise Exception()
                else:
                    average_rating = float(sum(score))/len(score)
                    writer.writerow({"doc1": doc1, "doc2": doc2,
                                     "average": average_rating})
