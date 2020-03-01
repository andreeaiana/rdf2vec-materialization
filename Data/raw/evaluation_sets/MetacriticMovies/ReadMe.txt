1. Description: The Metacritic Movies dataset is retrieved from Metacritic.com (http://www.metacritic.com/browse/movies/score/metascore/all), which contains an average rating of all time reviews for a list of movies. Each movie is linked to DBpedia using the movie's title and the movie's director. The initial dataset contained around 10,000 movies, from which we selected 1,000 movies from the top of the list, and 1,000 movies from the bottom of the list. We use the dataset both for regression as well as for classification, discretizing the target variable into ``good'' and ``bad'', using equal frequency binning.

2. ML taks: classificaiton and regression

3. Number of instances: 2000

4. Original source: http://www.metacritic.com/browse/movies/score/metascore/all

5. Linked to: DBpedia

6. Target variables
	-rating (Regression): discretized to "Label" (classification) by the rule: bad<40;good<infinity


7. Stratified data split (training/test):
	-label: TrainingSet.tsv (80%) and TestSet.tsv (20%)