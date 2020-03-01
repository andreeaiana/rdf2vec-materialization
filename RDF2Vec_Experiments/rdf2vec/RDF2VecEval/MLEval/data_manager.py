import os
import urllib
import pandas as pd
from gensim.models import Word2Vec
from SPARQLWrapper import SPARQLWrapper, JSON

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class data_manager:

    def __init__(self, gold_standard_file, vectors_file, w2v_model_name):
        self.gold_standard_file = gold_standard_file
        self.vectors_file = vectors_file
        self.w2v_model_name = w2v_model_name
        self.vector_size = int(
                self.w2v_model_name.split("_")[3].rsplit("v")[0])
        print("Data manager intialized.")

    def retrieve_vectors(self):
        if not os.path.isfile(self.vectors_file):
            print("Vectors not computed. Retrieving vectors.")
            gold = list(
                    self._read_file(
                            self.gold_standard_file, ["DBpedia_URI15"]
                            )["DBpedia_URI15"]
                    )
            if "DB" in self.w2v_model_name:
                processed_gold = [w.lstrip("http://dbpedia.org/resource/") for
                                  w in gold]
                processed_gold = [urllib.parse.quote(w, encoding="utf-8",
                                  safe=":/%#") for w in processed_gold]
                processed_gold = [w.replace("%C2%96", "%E2%80%93") if "%C2%96"
                                  in w else w for w in processed_gold]
                self._create_vectors(gold, processed_gold)
            else:
                processed_entities_list = list()
                entities = list()
                for entity in gold:
                    if "_ _" not in entity and " " not in entity and '"' not in entity:
                        wiki_entity = self._run_query(entity)
                        if not wiki_entity == "":
                                entities.append(entity)
                                processed_entity = wiki_entity.lstrip(
                                        "http://www.wikidata.org/entity/")
                                processed_entities_list.append(
                                        processed_entity)
                self._create_vectors(entities, processed_entities_list)
        print("Retrieving vectors.")
        vectors = self._read_vectors_file()
        return vectors

    def intersect_vectors_goldStandard(self, vectors, column_score,
                                       gold_standard_data=None,
                                       column_key="DBpedia_URI15"):
        gold = self._read_file(self.gold_standard_file, [column_key,
                                                         column_score])

        gold.rename(columns={column_key: "id"}, inplace=True)
        gold.rename(columns={column_score: column_score}, inplace=True)

        merged = pd.merge(gold, vectors, on="id", how="inner")
        output_left_merge = pd.merge(gold, vectors, how="outer",
                                     indicator=True)
        ignored = output_left_merge[output_left_merge["_merge"] == "left_only"]

        return merged, ignored

    def _create_vectors(self, gold, processed_gold):
        w2v_model = self._load_w2v_model()
        vectors_dict = dict()
        print("Creating vectors for the evaluation dataset.")
        for idx in range(len(gold)):
            if processed_gold[idx] in w2v_model.wv.vocab:
                vector = w2v_model.wv.get_vector(processed_gold[idx])
                vectors_dict[gold[idx]] = vector
        vectors = pd.DataFrame.from_dict(vectors_dict, orient="index")
        vectors.reset_index(level=0, inplace=True)
        print("Writing vectors to file.")
        vectors.to_csv(self.vectors_file, header=False, index=False,
                       encoding="latin1")
        print("Vectors created.")
        return None

    def _read_vectors_file(self):
        local_vectors = pd.read_csv(
                self.vectors_file,
                names=self._create_header(),
                encoding="latin1", index_col=False)
        return local_vectors

    def _create_header(self):
        headers = ["id"]
        for i in range(0, self.vector_size):
            headers.append(i)
        return headers

    def _read_file(self, filename, columns):
        return pd.read_csv(filename, "\t", usecols=columns, encoding="latin1")

    def _load_w2v_model(self):
        print("Loading Word2Vec model.")
        if "DB" in self.w2v_model_name:
            model_file = "../../../../Data/processed/models/DBpedia/" \
                + self.w2v_model_name
        else:
            model_file = "../../../../Data/processed/models/Wikidata/" \
                + self.w2v_model_name
        w2v_model = Word2Vec.load(model_file)
        print("Loaded.")
        return w2v_model

    def _run_query(self, dbpedia_instance):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        query = """
            PREFIX       owl:  <http://www.w3.org/2002/07/owl#>

            SELECT DISTINCT ?wikiEntity
            WHERE
              {""" + "<" + dbpedia_instance + ">" + \
            """
             owl:sameAs ?wikiEntity .
             FILTER regex(str(?wikiEntity), "wikidata.org/entity/Q") .
             }
            """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results_df = pd.io.json.json_normalize(results["results"]["bindings"])
        if not results_df.empty:
            wiki_entity = results_df[["wikiEntity.value"]]
            return wiki_entity["wikiEntity.value"][0]
        else:
            return ""
