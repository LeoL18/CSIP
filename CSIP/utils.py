import os
import pickle
from copy import deepcopy

import torch
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from numpy.random import RandomState
from PIL import Image
import matplotlib.pyplot as plt

from .dataset import GraphDataset
from .preprocessing import normalize_img

def show_image(image, label = None):
    '''
    Display an image with 5 channels using matplotlib.
    '''
    if len(image.shape) != 3:
        raise ValueError(f"image must have three dimensions, but your image has {len(image.shape)} dimensions")
    if image.shape[2] != 5:
        image = np.transpose(image, (1, 2, 0))
    fig, axs = plt.subplots(1, 5, figsize = (20, 8))
    for i in range(5): 
        axs[i].imshow(image[:, :, i], cmap = 'gray')
        axs[i].axis('off')
    fig.set_tight_layout('tight')
    if label is not None:
        axs[2].set_title(label)
    return axs

def group_path_by_channel(paths, channels = 5): # expect all images to come from the same folder
    grouped = [[] for i in range(channels)]
    paths = np.reshape(paths, (-1, 1))
    for path in paths:
        channel = int(os.path.basename(path[0])[15]) - 1
        grouped[channel].append(path[0])
    return np.array(grouped)

def get_pad(features, edges, pad = 160, size = 256):
    data_ = GraphDataset(
            features, edges, np.arange(len(edges)), idxs = np.arange(len(edges)), 
            pad = pad, size = size
        )
    features_, edges_, number_of_nodes_ = [], [], []
    for i in range(len(data_)):
        data_cur = data_[i]
        features_.append(data_cur[0])
        edges_.append(data_cur[1])
        number_of_nodes_.append(data_cur[3])
    return torch.stack(features_), edges_, torch.tensor(number_of_nodes_)

def unload_data(dir, load_graph = True, load_node = True, load_label = 2):
    """
    Given the path to the directory with all relevant files, unpack them and return them as objects. 
    This might take a while depending on the size of the files

    dir: the path to the directory
    sum_aggregate: whether we aggregate node features across channels
    load_label: 0 for do not load, 1 for labels only, 2 for everything

    Returns:
    (maps, graph, labels)
    maps: a list containing k dictionaries for k channels, 
    and each of those dictionaries contains one dictionary for each image, 
    and each of those dictionary contains the feature tensor of each cell in the image.

    graph: a list of nx objects containing the graphs for each image, respectively 

    labels: labels of the images. (well, pert, label)

    Note that you should name your files accordingly.
    
    """
    # nx graphs and nodes
    graph_path = os.path.join(dir, 'graphs_channel.pkl')
    nodes_path = os.path.join(dir, 'nodes.pth')

    # pert_name and smiles string
    perts_path = os.path.join(dir, 'perts.pkl')
    labels_path = os.path.join(dir, 'labels.pkl')

    # optional
    index_path = os.path.join(dir, 'idxs.pkl')
    wells_path = os.path.join(dir, 'wells.pkl')

    data = []
    if load_graph:
        print('Loading graphs...')
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        data.append(graph)
    if load_node:
        print('Loading nodes...')
        try: nodes = torch.load(nodes_path)
        except FileNotFoundError: 
            nodes_path = nodes_path[:-4] + '.pkl'
            with open(nodes_path, 'rb') as f: 
                nodes = pickle.load(f)
        data.append(nodes)
    if load_label > 0:
        
        print('Loading ground truth label...')

        if load_label == 2:

            with open(labels_path, 'rb') as f: labels = pickle.load(f)
            with open(perts_path, 'rb') as f: perts = pickle.load(f)
            data.append((labels, perts))

            try:
                with open(wells_path, 'rb') as f: wells = pickle.load(f)
                with open(index_path, 'rb') as f: index = pickle.load(f)
                data.append((wells, index))
            except FileNotFoundError:
                pass
        else:
            with open(labels_path, 'rb') as f: data.append(pickle.load(f))

    # features = []
    # channels = len(nodes)
    # for image_idx in range(len(nodes[0])):
    #     image_features = []
    #     for cell_idx in range(len(nodes[0][image_idx])):
    #         cell_features = torch.zeros_like(nodes[0][image_idx][cell_idx])
    #         for channel in range(channels):
    #             cell_features += nodes[channel][image_idx][cell_idx] # all cell in an image
    #         cell_features = cell_features.reshape(-1, )
    #         image_features.append(cell_features)
    #     features.append(image_features)
    # nodes = features

    return data

def illumination_correction(img): 
    pixels = img.flatten()
    pixels = sorted(pixels)
    q1, q3 = pixels[int(len(pixels)/4)], pixels[int(3*len(pixels)/4)]
    lb, ub = max(pixels[0], q1 - 1.3 * (q3 - q1)), min(pixels[len(pixels)-1], q3 + 1.3 * (q3 - q1))
    img = img.astype(np.float32)
    img = (img - lb) / (ub - lb)
    img[img < 0] = 0.0
    img[img > 1] = 1.0
    return img

def threshold_variance(img, thresh):

    return np.sum([mask.mean(dtype = torch.float32) * (img * mask).var()
                   for mask in [img >= thresh, img < thresh]])

# def analyze_G(G):
#     degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
#     dmax = max(degree_sequence)

#     fig = plt.figure("Degree of a random graph", figsize=(8, 8))
#     # Create a gridspec for adding subplots of different sizes
#     axgrid = fig.add_gridspec(5, 4)

#     ax0 = fig.add_subplot(axgrid[0:3, :])
#     Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
#     pos = nx.spring_layout(Gcc, seed=10396953)
#     nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
#     nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
#     ax0.set_title("Connected components of G")
#     ax0.set_axis_off()

#     ax1 = fig.add_subplot(axgrid[3:, :2])
#     ax1.plot(degree_sequence, "b-", marker="o")
#     ax1.set_title("Degree Rank Plot")
#     ax1.set_ylabel("Degree")
#     ax1.set_xlabel("Rank")

#     ax2 = fig.add_subplot(axgrid[3:, 2:])
#     ax2.bar(*np.unique(degree_sequence, return_counts=True))
#     ax2.set_title("Degree histogram")
#     ax2.set_xlabel("Degree")
#     ax2.set_ylabel("# of Nodes")

#     fig.tight_layout()
#     plt.show()

def reduce_bounding_boxes(centroids, img_rad = 32, overlap = 0.4, seed = 42):
    '''
    Reduge the number of bounding boxes based on the percentage of overlapping
    
    Args:
    centroids: (n, 2) ndarray, where n is the number of centroids
    img_rad: radius of image
    overlap: how much can two images overlap before one is removed
    seed: random seed for shuffling
    '''
    rs = RandomState(seed)
    rs.shuffle(centroids)
    filtered_centroids = []
    while len(centroids) > 0:
        centroid = centroids[-1]
        filtered_centroids.append(centroid)
        centroids = centroids[:-1]

        # finding the coordinates of the intersection box
        xmin = np.maximum(centroids[:,0] - img_rad, centroid[0] - img_rad)
        ymin = np.maximum(centroids[:,1] - img_rad, centroid[1] - img_rad)
        xmax = np.minimum(centroids[:,0] + img_rad, centroid[0] + img_rad)
        ymax = np.minimum(centroids[:,1] + img_rad, centroid[1] + img_rad)
        
        w = np.maximum(xmax - xmin, 0) # prevent negative numbers, which occur when there are no interection
        h = np.maximum(ymax - ymin, 0)

        I = w * h
        IoU = np.divide(I, (((2 * img_rad)**2) * 2 ) - I)
        masks = IoU < overlap
        centroids = centroids[masks]
    
    return np.array(filtered_centroids)

def split_images(img, split = 4, channel_first = False):
    # splits images into smaller images to augment dataset. split can be 4 or 1
    if split not in [1,4]: raise ValueError('Split must be either 1 or 4!')
    if split == 1: return np.expand_dims(img, 0)
    if channel_first: img = np.transpose(img, (1, 2, 0))
    h, w = img.shape[:2]
    h, w = int(h * 2 / split), int(w * 2 / split)
    img = img[:h * 2, :w * 2, :]
    split = np.array([img[h:, w:, :], img[h:, :w, :], img[:h, :w, :], img[:h, w:, :]]) # q1, q2, q3, q4
    if channel_first: split = np.transpose(split, (0, 3, 1, 2))
    return split

def split_centroids(centroids, split = 4, img_size = (1080, 1080)):
    # splits images into smaller images to augment dataset. split can be 4 or 1
    if split not in [1,4]: raise ValueError('Split must be either 1 or 4!')
    if split == 1: return np.expand_dims(centroids, 0)
    h, w = img_size
    h, w = int(h * 2 / split), int(w * 2 / split)
    q1, q2, q3, q4 = [], [], [], []
    for centroid in centroids:
        new = deepcopy(centroid)
        if centroid[0] >= h: new[0] -= h
        if centroid[1] >= w: new[1] -= w
        if centroid[0] < h:
            if centroid[1] < w: q3.append(new)
            else: q2.append(new)
        else:
            if centroid[1] < w: q4.append(new)
            else: q1.append(new)
    return [np.array(q1), np.array(q2), np.array(q3), np.array(q4)]

def reduce_bounding_boxes_on_images(centroids_list, img_rad = 32, overlap = 0.4):
    new_centroids_list = []
    for centroids in centroids_list: 
        new_centroids_list.append(reduce_bounding_boxes(centroids, img_rad, overlap))
    return np.array(new_centroids_list, dtype = object)

def encode(labels):
    return torch.tensor(OneHotEncoder().fit_transform(labels.reshape(-1,1)).todense())

def get_intensities(features):
    intensities = [torch.norm(feature) for feature in features]
    return intensities

def make_voroni_diagram(img_size, features, centroids, dpi = 144, save = False, save_path = None, save_name = 'voronoi'):

    vor = Voronoi(centroids)
    plt.figure(figsize = (img_size / dpi, img_size / dpi), dpi = dpi)
    
    fig = voronoi_plot_2d(vor, show_vertices = False, show_points = False)
    plt.axis('off')

    intensities = get_intensities(features)
    minima = min(intensities)
    maxima = max(intensities)

    norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Blues_r)

    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(intensities[r]))

    if save: 
        if save_path: plt.savefig(os.path.join(save_path, save_name) + '.pdf')
        else: plt.savefig(os.path.join(save_name) + '.pdf')
        
    plt.show()
    return vor

def plot_edges(ax, centroids, edge_view, color, deg):
    lines = []
    if deg is None:
        for edge in edge_view:
            u, v = edge
            lines.extend([(centroids[u - 1][0], centroids[v - 1][0]),
                        (centroids[u - 1][1], centroids[v - 1][1]), color])
    else:
        for edge in edge_view:
            u, v = edge
            c = color[u - 1] if deg[u - 1] > deg[v - 1] else color[v - 1]
            lines.extend([(centroids[u - 1][0], centroids[v - 1][0]),
                        (centroids[u - 1][1], centroids[v - 1][1]), c])
    ax.plot(*lines, linewidth = 0.4)

def get_graph_on_image(img_size, centroids, G, color = 'black', node_color = 'blue', size = 0.8, deg = None, dpi = 144, save = False, save_path = None, save_name = 'graph_image'):

    fig, ax = plt.subplots(figsize = (img_size / dpi, img_size / dpi), dpi = dpi)
    x, y = (*zip(*centroids,), )
    x, y = np.array(x), np.array(y)
    plot_edges(ax, centroids, G.edges(), color = color, deg = deg)
    plt.scatter(x, y, color = node_color, s = size)
    plt.axis('off')

    if save: 
        if save_path: plt.savefig(os.path.join(save_path, save_name) + '.pdf')
        else: plt.savefig(os.path.join(save_name) + '.pdf')

    return ax

def colored_k_hop_neighbor_hood(img_size, centroids, start_node, G, num_hops = 5, dpi = 144, save_path = None, save_name = 'graph_image'):

    fig, ax = plt.subplots(figsize = (img_size / dpi, img_size / dpi), dpi = dpi)
    x, y = (*zip(*centroids,), )
    x, y = np.array(x), np.array(y)
    plt.scatter(x, y, color = 'black', s = 0.3)
    hops = [-1 for i in range(G.number_of_nodes())]
    hops[start_node] = 0
    vis = [start_node]
    edges_hop = {}
    nodes_hop = {}
    while len(vis) > 0:
        cur = vis.pop(0)
        for edge in G.edges(cur):
            to = edge[0] if edge[0] != cur else edge[1]
            if hops[to - 1] == -1:
                hops[to - 1] = hops[cur - 1] + 1
                vis.append(to)

                if edges_hop[hops[to - 1]] is None: edges_hop[hops[to - 1]] = []
                edges_hop[hops[to - 1]].append((min(to, cur), max(to,cur)))
                if nodes_hop[hops[to - 1]] is None: nodes_hop[hops[to - 1]] = []
                nodes_hop[hops[to - 1]].append(to)

    plot_edges(ax, centroids, G.edges())
    
    for i in range(num_hops):  

        plt.scatter([centroids[nodes_hop[i][j] - 1][0] for j in range(len(nodes_hop[i]))],
                    [centroids[nodes_hop[i][j] - 1][1] for j in range(len(nodes_hop[i]))], color = 'r')
        plot_edges(ax, centroids, edges_hop[i], color = 'r')
            
        if save_path: plt.savefig(os.path.join(save_path, save_name)+ f'_{i}_hop.pdf')
        else: plt.savefig(os.path.join(save_name)+ f'_{i}_hop.pdf')
        
        # recovering for next diagram
        plt.scatter([centroids[nodes_hop[i][j] - 1][0] for j in range(len(nodes_hop[i]))],
                    [centroids[nodes_hop[i][j] - 1][1] for j in range(len(nodes_hop[i]))], color = 'black')
        plot_edges(ax, centroids, edges_hop[i])

def save_image_from_s3(s3, bucket, key, save_path, process = True):
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    with Image.open(file_stream) as im:
        if process:
            im = np.array(im)
            im = normalize_img(im)
            im = Image.fromarray(im)
        im.save(save_path)


def read_images_from_s3(s3, bucket, prefix):
    bucket = s3.Bucket(bucket)
    prefix_objs = bucket.objects.filter(Prefix=prefix)
    objects = []
    for obj in prefix_objs:
        key = obj.key
        body = Image.open(obj.get()['Body'])
        objects.append(np.array(body))
    return np.array(objects)

def pad_number(n):
    if n < 10:
        return '0' + str(n)
    else: return str(n)

def get_index(plate, row, col, site, idx_pair):
    return (plate - 1) * (
        idx_pair[0] * idx_pair[1] * idx_pair[2]) + (row - 1) * (
            idx_pair[1] * idx_pair[2]) + (col - 1) * (
                idx_pair[2]) + (site - 1)

def get_idx_from_path(path):
    row = path[1:3]
    col = path[4:6]
    site = path[7:9]
    return row, col, site

def get_path(platename, idx, ch, idx_pair, root = './'):
    '''
    Args:
    plate_name: path to the folder containing images from a plate (e.g. BR00116992__2020-11-05T21_31_31-Measurement1)
    image_idx: index of the image. All images are named in the format rXXcXXfXXp01-chXXsk1fk1fl1.tiff, where

    rXX is the row number of the well that was imaged. rXX ranges from r01 to r16.
    cXX is the column number of the well that was imaged. cXX ranges from c01 to c24.
    fXX corresponds to the site that was imaged. fXX ranges from f01 to f16.
    chXX corresponds to the fluorescent channels imaged. chXX ranges from ch01 to ch08

    then the image index is computed as idx = (row - 1) * 24 * 16 + (col - 1) * 16 + (site - 1), and all images
    can be uniquely identified as (plate, idx, ch). You can change the indexing by changing 'idx_pair'

    Note that for each plate there is a total of 49152 images, with idx running from 0 to 49151

    '''

    site = pad_number((idx % idx_pair[2]) + 1)
    col = pad_number(int((idx % (idx_pair[1] * idx_pair[2])) / 24) + 1)
    row = pad_number(int(idx / (idx_pair[1] * idx_pair[2])) + 1)
    ch = str(ch + 1)

    path = f'r{row}c{col}f{site}p01-ch{ch}sk1fk1fl1.tiff'

    return os.path.join(root, platename, 'Images', path)
