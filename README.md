# rdf2vec-materialization
The source code for the paper "More is not Always Better: The Negative Impact of A-box Materialization on RDF2vec Knowledge Graph Embeddings"


# Running the code

## Data
To run the experiments, the 2016 DBPedia dump was used. The required files (instance_types_transitive_en.ttl.bz2, mappingbased_objects_en.ttl.bz2, dbpedia_2016-10.nt) can be downloaded from https://wiki.dbpedia.org/develop/datasets/dbpedia-version-2016-10).
The raw datasets are stored in `Data/raw/DBpedia`.

## Enriching DBpedia
To strategies were used to materialize subproperties, inverse, symmetric and transitive properties.

* Map DBpedia properties to Wikidata, using the DBPedia_enrichment.ipynb Jupyter notebook. The properties used in the experiments an be loaded from `Data/interim/properties/DBpedia_Wikidata_mapped_<property_type>_prop.csv`.
* Mine symmetric, transitive and inverse properties from DBpedia with the DL_Learner tool (version 1.3.0) using the DL-Learner_enrichment.ipynb. Enrich the original graph using the DBPedia_enrichment (DLLearner).ipynb Jupyter notebook. The used properties can be loaded from `Data/interim/properties/DBpedia_dllearner_<property_type>_properties.csv`. 


## Computing the random walks
To compute the random walks, the KGvec2go Walks code (https://github.com/janothan/kgvec2go-walks) was used on the original and enriched DBpedia graphs.
Firstly, extract the list of entities using the "entities_extractor.py" script.
Secondly, build the KGvec2go Walks project in the command line (mvn package) and start the jar with the command -guided or -help.
The resulting random walks are stored under `Data/interim/DBpedia/`

## Training the RDF2Vec model
To train the RDF2VEc model, run the run_w2v.py script on the computed random walks. This will compute 2 models, with vectors of dimension 500 and 200, respectively.
The models are saved under: `Data/processed/models/DB2Vec_<#walks>w_<depth>d_<vector_dimension>v_<graph_type>`

## Evaluation
To evaluate the models, run the evaluator.py scripts for the different evaluation tasks under RDF2VecEval.
The evaluation vectors are stored under `Data/processed/evaluation_vectors/DB2Vec_<#walks>w_<depth>d_<vector_dimension>v_<graph_type>` and the results under `Data/processed/evaluation_results/DB2Vec_<#walks>w_<depth>d_<vector_dimension>v_<graph_type>`.
