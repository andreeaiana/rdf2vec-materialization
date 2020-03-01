# -*- coding: utf-8 -*-
import os
import io
import time
import gzip
import urllib
import argparse
from tqdm import tqdm
from rdflib import Graph, URIRef, BNode, Literal
from graph import rdflib_to_kg, extract_instance


class Extractor:

    def __init__(self, graph_name, graph_type):
        self.graph_name = graph_name
        self.graph_type = graph_type

        if self.graph_type == "original":
            self.graph_type_name = ""
        else:
            self.graph_type_name = "_" + self.graph_type

    def extract_entities(self):
        # Load file into graph
        graph = self._load_graph()

        # Convert to knowledge graph
        print("Converting to knowledge graph.")
        t = time.time()
        self.kg = rdflib_to_kg(graph)
        print("Knwoledge graph created.")
        print('Time to create knowledge graph: {} mins\n\n'.format(
                round((time.time() - t) / 60, 2)))

        # Extracting list of entities
        print("Extracting list of entities.")
        t = time.time()
        entities = list(set(graph.subjects()))
        print("Entities extracted. Extracted {} entities.".format(
                len(entities)))
        print('Time to extract entities: {} mins\n\n'.format(
                round((time.time() - t) / 60, 2)))
        del graph

        print("Saving entities to disk...")
        entities_file = "../../../Data/interim/" + self.graph_name \
                        + "/graph/" + str.lower(self.graph_name) + "_graph" \
                        + self.graph_type_name + "_labels_en.txt"
        with open(entities_file, "w", encoding="utf-8") as f:
            for entity in entities:
                f.write("%s\n" % entity)
        print("Saved.")

    def _load_graph(self):
        graph_file = "../../../Data/interim/" + self.graph_name + "/graph/" \
            + str.lower(self.graph_name) + "_graph" + self.graph_type_name \
            + ".nt.gz"

        # Load file into graph
        print("Reading file and creating graph.")
        t = time.time()
        graph = Graph()

        print("Computing file size.")
        with gzip.open(graph_file, 'rt') as file_obj:
            file_size = file_obj.seek(0, io.SEEK_END)
        print("File size in bytes: {}.".format(file_size))

        lines_skipped = 0
        with tqdm(desc="Reading triples and adding to graph: ",
                  total=file_size, unit="bytes") as pbar:
            with gzip.open(filename=graph_file, encoding="utf-8",
                           mode="rt") as f:
                for line in f:
                    if line not in ['\n', '\r\n']:
                        try:
                            triples = line.split(" ", maxsplit=2)
                            subj = self._create_node(triples[0])
                            pred = self._create_node(triples[1])
                            obj = self._create_node(triples[2].rsplit(" .")[0])
                            graph.add((subj, pred, obj))
                        except Exception as e:
                            print(line, e)
                            lines_skipped += 1
                    pbar.update(len(line))
        print("Lines skipped: {}.".format(str(lines_skipped)))
        print('Time to create graph: {} mins\n\n'.format(
                round((time.time() - t) / 60, 2)))
        print("Triples in graph after add: {}.\n\n".format(len(graph)))
        return graph


    def _quote(self, string):
        return urllib.parse.quote(string, encoding="utf-8", safe=":/%#")

    def _create_node(self, string):
        if string.startswith("<"):
            if string[-2] != "/":
                return URIRef(self._quote(string[1:-1]))
            else:
                return URIRef(self._quote(string[1:-2]))
        elif string.startswith('"'):
            return Literal(string[1:string.rindex('"')])
        elif string.startswith("_:"):
            return BNode(string[2:])
        else:
            return "Invalid node type"

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                description='Arguments for Word2Ved model.')
        parser.add_argument('graph_name',
                            choices=["DBpedia", "Wikidata"],
                            help='the name of the dataset used for training')
        parser.add_argument("graph_type",
                            choices=["original", "enriched",
                                     "enriched_dllearner"],
                            help="the dataset type")
        args = parser.parse_args()

        print("Starting...\n")
        from entities_extractor import Extractor
        extractor = Extractor(args.graph_name, args.graph_type)
        extractor.extract_entities()
        print("Finished.")
