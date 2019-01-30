# -*- coding:utf-8 -*-
import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec")

    parser.add_argument('--input',nargs='?',default='graph/Graph_1960-2015_A63F_5-00_game-pachinko_processed.dot',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/patent.emb',
                        help='Embeddings path')

    parser.add_argument('--output_vec', nargs='?', default='emb/vector.tsv',
                        help='vector.tsv path')

    parser.add_argument('--output_meta', nargs='?', default='emb/metadata.tsv',
                        help='metadata.tsv path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of vec\'s dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    return parser.parse_args()

def read_graph():
    # load citation network graph
    G = nx.read_edgelist(args.input, nodetype = int, create_using = nx.DiGraph())

    # initialize weight
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    # Undirected graph
    G = G.to_undirected()
    return G

def learn_embeddings(walks):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)

    return model


def save_embedding_projector_files(model, vector_file, metadata_file):
    with open(vector_file, 'w', encoding='utf-8') as f, \
         open(metadata_file, 'w', encoding='utf-8') as g:

        for word in model.wv.vocab.keys():
            embedding = model.wv[word]

            # Save vector TSV file
            f.write('\t'.join([('%f' % x) for x in embedding]) + '\n')

            # Save metadata TSV file
            g.write(word + '\n')
    f.close()
    g.close()
    return


def main(args):
    start = time.time()
    np.random.seed(0)

    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learn_embeddings(walks)
    save_embedding_projector_files(model=model, vector_file=args.output_vec,metadata_file=args.output_meta)

    print("elapsed time:{0}".format(time.time()-start) + "[sec]")

if __name__ == "__main__":
    args = parse_args()
    main(args)
