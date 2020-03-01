1. Description: Similarly as the Metacritic Movies, the Metacritic Albums dataset is retrieved from Metacritic.com (http://www.metacritic.com/browse/albums/score/metascore/all), which contains an average rating of all time reviews for a list of albums. Each album is linked to DBpedia using the album's title and the album's artist.

2. ML taks: classificaiton and regression

3. Number of instances: 1600

4. Original source: http://www.metacritic.com/browse/albums/score/metascore/all

5. Linked to: DBpedia

6. Target variables
	-rating (Regression): discretized to "label" (classification) by the rule: bad<63;good<infinity


7. Stratified data split (training/test):
	-label: TrainingSet.tsv (70%) and TestSet.tsv (30%)
