import os
import urllib
import codecs
import pandas as pd
from gensim.models import Word2Vec
from SPARQLWrapper import SPARQLWrapper, JSON

# DISCLAIMER
# File modified from https://github.com/mariaangelapellegrino/Evaluation-Framework


class data_manager:

    def __init__(self, vectors_file_input, vectors_file_output,
                 w2v_model_name):
        self.vectors_file_input = vectors_file_input
        self.vectors_file_output = vectors_file_output
        self.w2v_model_name = w2v_model_name
        self.vector_size = int(
                self.w2v_model_name.split("_")[3].rsplit("v")[0])
        print("Data manager intialized.")

    def retrieve_vectors(self):
        if not os.path.isfile(self.vectors_file_input) or not os.path.isfile(self.vectors_file_output):
            print("Vectors not computed. Retrieving input and output vectors.")
            gold = self.read_file()
            entities_list = list()
            for main_entity, related_entities in gold.items():
                entities_list.append(main_entity)
                entities_list.extend(related_entities)
            if "DB" in self.w2v_model_name:
                processed_entities_list = [w.lstrip(
                        "http://dbpedia.org/resource/") for w in entities_list]
                processed_entities_list = [
                        urllib.parse.quote(w, encoding="utf-8", safe=":/%#")
                        for w in processed_entities_list]
                processed_entities_list = [
                        w.replace("%C2%96", "%E2%80%93") if "%C2%96" in w
                        else w for w in processed_entities_list]
                self._create_input_vectors(entities_list,
                                           processed_entities_list)
                self._create_output_vectors(entities_list,
                                            processed_entities_list)
            else:
                processed_entities_list = list()
                entities = list()
                for entity in entities_list:
                    if '"' not in entity:
                        wiki_entity = self._run_query(entity)
                        if not wiki_entity == "":
                            entities.append(entity)
                            processed_entity = wiki_entity.lstrip(
                                        "http://www.wikidata.org/entity/")
                            processed_entities_list.append(processed_entity)
                self._create_input_vectors(entities, processed_entities_list)
                self._create_output_vectors(entities, processed_entities_list)
        print("Retrieving vectors.")
        input_vectors = self._read_vectors_file(self.vectors_file_input)
        output_vectors = self._read_vectors_file(self.vectors_file_output)
        return input_vectors, output_vectors

    def intersect_vectors_goldStandard(self, vectors, gold_standard_data=None,
                                       column_key=None, column_score=None):
        if gold_standard_data is None:
            entities = self.read_file()
            gold_standard_data = pd.DataFrame({"id": entities.keys()})

        merged = pd.merge(gold_standard_data, vectors, on="id", how="inner")
        output_left_merge = pd.merge(gold_standard_data, vectors, on="id",
                                     how="outer", indicator=True)
        ignored = output_left_merge[output_left_merge["_merge"] == "left_only"]

        return merged, ignored

    def _create_input_vectors(self, entities_list, processed_entities_list):
        w2v_model = self._load_w2v_model()
        vectors_dict = dict()
        print("Creating vectors for the evaluation dataset.")
        for idx in range(len(entities_list)):
            if processed_entities_list[idx] in w2v_model.wv.vocab:
                entity = w2v_model.wv.vocab[processed_entities_list[idx]]
                vector = w2v_model.wv.vectors[entity.index]
                vectors_dict[entities_list[idx]] = vector
        vectors = pd.DataFrame.from_dict(vectors_dict, orient="index")
        vectors.reset_index(level=0, inplace=True)
        print("Writing vectors to file.")
        vectors.to_csv(self.vectors_file_input, header=False, index=False,
                       encoding="latin1")
        print("Vectors created.")
        return None

    def _create_output_vectors(self, entities_list, processed_entities_list):
        w2v_model = self._load_w2v_model()
        vectors_dict = dict()
        print("Creating vectors for the evaluation dataset.")
        for idx in range(len(entities_list)):
            if processed_entities_list[idx] in w2v_model.wv.vocab:
                entity = w2v_model.wv.vocab[processed_entities_list[idx]]
                vector = w2v_model.trainables.syn1neg[entity.index]
                vectors_dict[entities_list[idx]] = vector
        vectors = pd.DataFrame.from_dict(vectors_dict, orient="index")
        vectors.reset_index(level=0, inplace=True)
        print("Writing vectors to file.")
        vectors.to_csv(self.vectors_file_output, header=False, index=False,
                       encoding="latin1")
        print("Vectors created.")
        return None

    def _read_vectors_file(self, file):
        local_vectors = pd.read_csv(file, names=self._create_header(),
                                    encoding="latin1", index_col=False)
        return local_vectors

    def _create_header(self):
        headers = ["id"]
        for i in range(0, self.vector_size):
            headers.append(i)
        return headers

    def read_file(self):
        entities_groups = {}
        related_entities = []

        gold_standard_file = "../../../../Data/raw/evaluation_sets/" \
            + "KORE_entity_relatedness/KORE.txt"
        f = codecs.open(gold_standard_file, "r", "utf-8")

        for i, line in enumerate(f):
            key = line.rstrip().lstrip()

            if i % 21 == 0:
                main_entity = key
                related_entities = []
            else:
                related_entities.append(key)

            if i % 21 == 20:
                entities_groups[main_entity] = related_entities

        return entities_groups

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
