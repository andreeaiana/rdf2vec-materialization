1. Description: The Cities dataset contains a list of cities and their quality of living (as a numerical score), as captured by Mercer. The cities are linked to DBpedia. We use the dataset both for regression as well as for classification, discretizing the target variable into ``high'', ``medium'', and ``low'', using equal frequency binning.

2.ML taks: classificaiton and regression

3. Number of instances: 212

4. Original source: http://across.co.nz/qualityofliving.htm

5. Linked to: DBpedia

6. Target variables
	-rating (Regression): discretized to "label" (classification) by the rule: low<50;medium<96;high<infinity


7. Stratified data split (training/test):
	-label: TrainingSet.tsv (80%) and TestSet.tsv (20%)
	