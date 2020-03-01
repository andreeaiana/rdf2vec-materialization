|--------------------------------|
| KORE: Related Entities Dataset |
|--------------------------------|

2012-07-18
Johannes Hoffart

Overview
--------
This dataset was created for [1] to compare different measures of semantic relatedness between entities. It contains 21 seed entities from various domains (IT companies, TV series, Hollywood celebrities, and Video Games). For each of the seeds it contains 20 entities ranked by their relatedness to the seed, with the most related one ranked first. The 20 entities were chosen from the Wikipedia articles of the seed entities so that their is some variance in the relatedness to the seed. The ranking was then created by doing pairwise comparisons of the related entity candidates on crowdflower, aggregating the pairwise comparisons according to [2]. For details on how the dataset was created, please see [1].


Format
------
The file rankedrelatedentities.txt contains the data in the following format

seed entity A
<TAB>entity most related to A
<TAB>entity second most related to A
...
<TAB>entity least related to A
seed entity B
<TAB>entity most related to B
...

The names of the entities are the titles of the Wikipedia artciles describing them. The title is from the 2010-08-17 dump of the English Wikipedia, which was also used to create the initial version of YAGO2 [3]. The titles may change over time, however it should be easy enough to adapt the names to newer versions of Wikipedia by just replacing the entity name with the more recent one.


License
-------

The KORE: related entities dataset is licensed under the Creative Commons Attribution 3.0 License (http://creativecommons.org/licenses/by/3.0/)


References
----------

[1] J. Hoffart, S. Seufert, D. B. Nguyen, M. Theobald, and G. Weikum, “KORE: Keyphrase Overlap Relatedness for Entity Disambiguation,” presented at the Proceedings of the 21set ACM International Conference on Information and Knowledge Management, CIKM 2012, Hawaii, USA, 2012.
[2] D. Coppersmith, L. K. Fleischer, and A. Rurda, “Ordering by weighted number of wins gives a good ranking for weighted tournaments,” Transactions on Algorithms, vol. 6, no. 3, 2010.
[3] J. Hoffart, F. M. Suchanek, K. Berberich, and G. Weikum, “YAGO2: A spatially and temporally enhanced knowledge base from Wikipedia,” Artificial Intelligence, 2012.