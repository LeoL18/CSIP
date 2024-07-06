import pickle
from CSIP.utils import unload_data
import os
from tqdm import tqdm
from argparse import ArgumentParser

def pad(x):
    x = str(x)
    while len(x) < 6:
        x = '0' + x
    return x

def __main__():

    parser = ArgumentParser()
    parser.add_argument("--data_path", type = str, default = None, help = "Path to processed data")
    parser.add_argument("--save_feature", type = str, default = None, help = "Path to save unwrapped dataset")
    parser.add_argument("--save_label", type = str, default = None, help = "Path to save labels")

    args = parser.parse_args()

    graph, nodes, labels = unload_data(args.data_path, load_label = 1)
    with open(args.save_label, 'wb') as file:
        pickle.dump(labels, file)
        
    for i in range(len(nodes)):
        nodes[i] = nodes[i][0]
    for i in range(len(graph)):
        graph[i] = graph[i][0]

    l = 1
    units = int(len(nodes) / l)

    for i in tqdm(range(units)):

        start, end = l * i, l * (i + 1)
        if l!=1:
            path_g = os.path.join(args.save_feature, f'graph_{pad(start)}-{pad(end - 1)}')
            path_f = os.path.join(args.save_feature, f'feature_{pad(start)}-{pad(end - 1)}')
            g = graph[start: end]
            f = nodes[start: end]
        else:
            path_g = os.path.join(args.save_feature, f'graph_{pad(start)}.pkl')
            path_f = os.path.join(args.save_feature, f'feature_{pad(start)}.pth')
            g = graph[start]
            f = nodes[start]

        with open(path_g, 'wb') as file:
            pickle.dump(g, file)
        with open(path_f, 'wb') as file:
            pickle.dump(f, file)