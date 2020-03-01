# -*- coding: utf-8 -*-
import os
import io
import time
import gzip
import urllib
import logging
from tqdm import tqdm
from gensim.models import Word2Vec


class SentencesGenerator(object):
    def __init__(self, walks_folder):
        self.walks_folder = walks_folder

    def __iter__(self):
        with tqdm(desc="Generating sentences: ", total=len(
                os.listdir(self.walks_folder))) as pbar:
            for fname in os.listdir(self.walks_folder):
                try:
                    for line in gzip.open(
                            os.path.join(self.walks_folder, fname), mode='rt'):
                        if not line == "\n":
                            words = line.split()
                            words_processed = []
                            for word in words:
                                if "dbr:" in word:
                                    word = word.split("dbr:")[-1]
                                else:
                                    word = word.split("/")[-1]
                                words_processed.append(word)
                            yield words_processed
                except Exception:
                    print("Failed reading file:")
                    print(fname)
                pbar.update(1)


class W2VModel:

    def __init__(self, graph_name, graph_type, walks_per_graph, walks_depth):
        self.graph_name = graph_name
        self.graph_type = graph_type
        self.walks_per_graph = walks_per_graph
        self.walks_depth = walks_depth
        self.walks_folder = self._get_walks_folder()

        if self.graph_type == "original":
            self.graph_type_name = ""
        else:
            self.graph_type_name = "_" + self.graph_type

        log_file = "../../../Data/processed/models/" + self.graph_name + "/" \
            + "log_file" + self.graph_type_name + "_" \
            + str(self.walks_per_graph) + "w_" + str(self.walks_depth) \
            + "d.out"
        logging.basicConfig(format="%(asctime)s - %(levelname)s : %(message)s",
                            filename=log_file, datefmt='%H:%M:%S',
                            level=logging.INFO)
        print("RDF2Vec model initialized.")

    def train(self):
        # Compute sentences
        print("Training models.")
        sentences = SentencesGenerator(self.walks_folder)

        # sg 500
        print("MODEL 1: SG 500v\n")
        model = Word2Vec(size=500, workers=20, window=5, sg=1, negative=25,
                         iter=5, min_count=1)

        # Build vocabulary
        print("\tBuilding vocabulary.")
        t = time.time()
        model.build_vocab(sentences, progress_per=10000)
        print("\tVocabulary built.")
        print('\tTime to build vocab: {} mins\n'.format(round((
                time.time() - t) / 60, 2)))

        # Train model
        print("\tTraining model.")
        t = time.time()
        model.train(sentences, total_examples=model.corpus_count,
                    epochs=model.epochs)
        print("\tModel trained.")
        print('\tTime to build vocab: {} mins\n'.format(round((
                time.time() - t) / 60, 2)))

        # Save model
        self._save_model(model, 500)

        # sg 200
        print("MODEL 2: SG 200v\n")
        model2 = Word2Vec(size=200, workers=20, window=5, sg=1, negative=25,
                          iter=5, min_count=1)
        model2.reset_from(model)
        del model

        # Train model
        print("\tTraining model.")
        t = time.time()
        model2.train(sentences, total_examples=model2.corpus_count,
                     epochs=model2.epochs)
        print("\tModel trained.")
        print('\tTime to build vocab: {} mins\n'.format(
                round((time.time() - t) / 60, 2)))

        # Save model
        self._save_model(model2, 200)
        del model2

        print("Finished training models.")

    def _get_walks_folder(self):
        if self.graph_name == "DBpedia":
            if self.graph_type == "original":
                walks_folder = "../../../Data/interim/DBpedia/DBpediaWalks/" \
                    + str(self.walks_per_graph) + "w_" \
                    + str(self.walks_depth) + "d/"
            elif self.graph_type == "enriched":
                walks_folder = "../../../Data/interim/DBpedia/" \
                    + "DBpediaEnrichedWalks/" + str(self.walks_per_graph) \
                    + "w_" + str(self.walks_depth) + "d/"
            else:   # self.graph_type == "enriched_dllearner"
                walks_folder = "../../../Data/interim/DBpedia/" \
                    + "DBpediaEnrichedWalks_DLLearner/" \
                    + str(self.walks_per_graph) + "w_" \
                    + str(self.walks_depth) + "d/"
        else:
            if self.graph_type == "original":
                walks_folder = "../../../Data/interim/Wikidata/" \
                    + "WikidataWalks/" + str(self.walks_per_graph) + "w_" \
                    + str(self.walks_depth) + "d/"
            else:  # self.graph_type == "enriched"
                walks_folder = "../../../Data/interim/Wikidata/" \
                    + "WikidataEnrichedWalks/" + str(self.walks_per_graph) \
                    + "w_" + str(self.walks_depth) + "d/"
        return walks_folder

    def _save_model(self, model, vector_size):
        if self.graph_name == "DBpedia":
            model_file = "../../../Data/processed/models/DBpedia/DB2Vec_" \
                + str(self.walks_per_graph) + "w_" + str(self.walks_depth) \
                + "d_" + str(vector_size) + "v" + self.graph_type_name
        else:
            model_file = "../../../Data/processed/models/Wikidata/WD2Vec_" \
                + str(self.walks_per_graph) + "w_" + str(self.walks_depth) \
                + "d_" + str(vector_size) + "v" + self.graph_type_name
        print("\tSaving model.")
        model.save(model_file)
        print("\tModel saved.\n")

    def main():
        import argparse
        parser = argparse.ArgumentParser(
                description='Arguments for Word2Ved model.')
        parser.add_argument('graph_name',
                            choices=["DBpedia", "Wikidata"],
                            help='the name of the dataset used for training')
        parser.add_argument("graph_type",
                            choices=["original", "enriched",
                                     "enriched_dllearner"],
                            help="the dataset type")
        parser.add_argument('walks_per_graph',
                            choices=[200, 500],
                            type=int,
                            help='random walks iterations')
        parser.add_argument('walks_depth',
                            choices=[2, 4, 8],
                            type=int,
                            help='random walks depth')
        args = parser.parse_args()
        from run_w2v import W2VModel
        model = W2VModel(args.graph_name, args.graph_type,
                         args.walks_per_graph, args.walks_depth)
        model.train()

    if __name__ == "__main__":
        main()
