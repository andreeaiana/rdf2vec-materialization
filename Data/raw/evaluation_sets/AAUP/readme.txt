1.Description: The AAUP (American Association of University Professors) dataset contains a list of universities, including eight target variables describing the average salary and the average compensation of different staff at the universities (http://www.amstat.org/publications/jse/jse_data_archive.htm). We use the average salary, and average compensation, as a target variable both for regression as well as for classification, discretizing the target variable into ``high'', ``medium'', and ``low'', using equal frequency binning. Each university in the dataset was linked to the corresponding resource in DBpedia based on the university's name.

2.ML taks: classificaiton and regression

3. Number of instances: 960

4. Original source: JSE (http://www.amstat.org/publications/jse/jse_data_archive.htm)

5. Linked to: DBpedia

6. Number of existing feature: 17

7. Target variables
	-Average_salary_all_ranks (Regression): discretized to "label_salary" (classification) by the rule: low<350;medium<500;high<infinity
	-Average_compensation_all_ranks (Regression): discretized to "label_comp" (classification) by the rule: low<445;medium<595;high<infinity


8. Stratified data split (training/test):
	-label_salary: TrainingSet(salary).tsv (70%) and TestSet(salry).tsv (30%)
	-label_comp: TrainingSet(comp).tsv (70%) and TestSet(comp).tsv (30%)