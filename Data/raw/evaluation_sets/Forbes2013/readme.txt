The Forbes dataset contains a list of companies including several features of the companies (such as sales, profits, assets), which was generated from the Forbes list of leading companies 2015 (http://www.forbes.com/global2000/list/). The target is to predict the company's market value as a regression task. To use it for the task of classification we discretize the target variable into ``high'', ``medium'', and ``low'', using equal frequency binning. Each company in the dataset was linked to the corresponding resource in DBpedia based on the company's name.

2.ML taks: classificaiton and regression

3. Number of instances: 1585

4. Original source: http://www.forbes.com/global2000/list/

5. Linked to: DBpedia

6. Number of existing feature: 8

7. Target variables
	-Market_value (Regression): discretized to "label" (classification) by the rule: low<10;medium<90;high<infinity


8. Stratified data split (training/test):
	-Market_value: TrainingSet.tsv (70%) and TestSet.tsv (30%)
	