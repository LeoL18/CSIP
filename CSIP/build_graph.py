import pickle
import os
from glob import glob
import warnings
import concurrent
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import PIL
from PIL import Image
from tqdm import tqdm
import networkx as nx
from torch.utils.data import SequentialSampler

from .utils import group_path_by_channel, split_centroids, split_images
from .feature_extraction import ConvExtractor, InterpolationExtractor
from .preprocessing import normalize_img

warnings.filterwarnings('ignore')

class ImageDataset(Dataset):

    def __init__(self, root, centroids, channels = 5, 
                 imgs = None, normalize = True, illum_which = None, illum = None):
        
        '''
        Dataset for training CNN autoencoder

        root: path to image directory, which should only contain relevant images
        centroids: nested list containing centroids for each cell in the order (img, cell, coord)
        channels: how many channels the images have
        imgs: optional, images can be directly provided
        normalize: whether we normalize the images
        illum_which: optional, a list identifying the illumination correction mask each image corresponds to
        illum: list of illumination correction masks


        '''
        
        self.root = root
        self.channels = channels
        self.centroids = centroids 
        self.use_imgs = False
        self.normalize = normalize
        self.illum_which = illum_which
        self.illum = illum
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        if imgs is not None:
            self.imgs = imgs
            self.use_imgs = True
        
        self.paths = glob(os.path.join(self.root, '*'))
        

    def __len__(self):
        if self.use_imgs: return self.imgs.shape[0]
        else: return len(self.paths[0])
    
    def __getitem__(self, i): ######
        if self.use_imgs: img = self.imgs[i]
        else:
            img = []
            try:
                for j in range(self.channels): 
                    img.append(np.array(Image.open(
                        os.path.join(self.root, self.paths[j][i]))))
            # except OSError:
            #     for j in range(self.channels): 
            #         img.append(np.array(Image.open(
            #             os.path.join(self.root, self.paths[j][i]))))
                    
            except PIL.UnidentifiedImageError:
                return None
                
        img = np.array(img, dtype = np.int32)

        if self.normalize:
            img = normalize_img(img)

        img = torch.tensor(img, dtype = torch.float32) / 255.
        if self.illum_which is not None:
            img = img / self.illum[self.illum_which[i]]

        return i, img, self.centroids[i]
    
class ImageLoader:

    def __init__(self, datasets, img_rad = 34, batch_size = 32, mask_proportion = 0.2, mask_size = 0.4):
        '''
        ImageLoader
        loader for training CNN autoencoder

        dataset: ImageDataset
        img_rad: size of images to use
        batch_size: batch size
        mask_proportion: how much of the image to apply mask on
        mask_size: how much of the image we mask out

        Mask helps the CNN extractor to learn extracing information from incomplete cell images.
        if mask_proportion is nonzero, loader additionally returns the unmasked images as training labels
        '''
        
        self.img_rad = img_rad
        self.centroids = []
        self.datasets = datasets
        self.cur = 0
        self.batch_size = batch_size
        self.mask_proportion = mask_proportion
        self.mask_size = mask_size
        centroids = []
        for idx, img, centroids_img in tqdm(datasets, desc = 'generating dataset'):
            idxs = (centroids_img - self.img_rad < 0).sum(axis = 1) + (centroids_img + self.img_rad > img.shape[2]).sum(axis = 1)
            idxs = np.nonzero(idxs == 0)[0]
            centroids_img = centroids_img[idxs]
            l = centroids_img.shape[0]
            centroids.extend(zip(centroids_img.tolist(), [idx] * l))
        self.centroids.extend(centroids)
        # self.centroids = np.array(self.centroids)
        np.random.shuffle(self.centroids)
    
    def __len__(self): return len(self.centroids)
    def __getitem__(self, i):
        centroid, idx = self.centroids[i]
        img = self.datasets[idx][1]
        centroid[0], centroid[1] = int(centroid[0]), int(centroid[1])
        cropped = img[:, centroid[1] - self.img_rad : centroid[1] + self.img_rad, centroid[0] - self.img_rad: centroid[0] + self.img_rad]
        cropped = cropped.to(torch.float)
        # cropped = (cropped - torch.min(cropped)) / (torch.max(cropped) - torch.min(cropped))
        if self.mask_proportion == 0: return cropped

        mask = np.random.rand(1)[0] <= self.mask_proportion
        original = cropped
        if mask:
            expand = max(self.mask_size * self.img_rad + 
                         np.random.randint(-int(0.15 * self.img_rad), int(0.15 * self.img_rad)), 0)
            expand = round(expand)
            choose = np.random.randint(0,4)
            h, w = cropped.shape[1: 3]
            if choose == 0: cropped[:, :, :expand] = 0
            elif choose == 1: cropped[:, :, w - expand:] = 0
            elif choose == 2: cropped[:, :expand, :] = 0
            else: cropped[:, h - expand:, :] = 0

        return (cropped, original)
    def __iter__(self): 
        self.cur = 0
        return self
    def __next__(self):
        if self.cur >= len(self) - 1: self.cur = 0 # restart

        masked, original = [], []

        if self.mask_proportion == 0:
            for i in range(self.cur, min(len(self), self.cur + self.batch_size)):
                original.append(torch.unsqueeze(self[i], 0))

            self.cur += self.batch_size
            return torch.cat(original)
        
        for i in range(self.cur, min(len(self), self.cur + self.batch_size)):
            mask, ori = self[i]
            masked.append(torch.unsqueeze(mask, 0))
            original.append(torch.unsqueeze(ori, 0))
            
        self.cur += self.batch_size
        return torch.cat(masked), torch.cat(original)

class FeatureExtractor: 

    def __init__(self, device = 'cpu', img_size = 34, min_node = 15, mode = 'cnn', 
                 split = 4, load = True, load_path = None, num_workers = 4):
        
        '''
        
        '''
        models = {
            'cnn': ConvExtractor(),
            'nearest': InterpolationExtractor('nearest'),
            'linear': InterpolationExtractor('linear'),
            'bilinear': InterpolationExtractor('bilinear'),
            'bicubic': InterpolationExtractor('bicubic'),
            'trilinear': InterpolationExtractor('trilinear'),
        }
        self.model = models[mode]
        if load: self.model.load_state_dict(torch.load(load_path)['model_state'])
        self.preprocess = torchvision.transforms.ToTensor()
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.img_size = img_size
        self.min_node = min_node
        self.split = split
        self.num_workers = num_workers

    def pad_img(self, centroid, img):
        x, y = centroid[0].item(), centroid[1].item()
        if (centroid - self.img_size < 0).sum() + (centroid + self.img_size > img.shape[1]).sum() > 0: 
            padded = torch.zeros(img.shape[0], 2 * self.img_size, 2 * self.img_size, 
                                 dtype = torch.float32)
            
            x1 = max(x - self.img_size, 0)
            x2 = min(x + self.img_size, img.shape[2])
            y1 = max(y - self.img_size, 0)
            y2 = min(y + self.img_size, img.shape[1])
            padded[:, y1 - y + self.img_size 
                   : y2 - y + self.img_size, x1 - x + self.img_size
                   : x2 - x + self.img_size] = torch.tensor(img[:, y1:y2, x1:x2]) # keep what we have, pad the rest with 0
            return padded
        else:
            return img[:, y - self.img_size :  #########
                       y + self.img_size, x - self.img_size: x + self.img_size]

    def get_feature_img(self, img):
        img = torch.unsqueeze(img, 0).to(self.device)
        if type(img) != type(torch.tensor(0)): img = self.preprocess(img)
        img = img.to(torch.float32)
        # img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        if img.shape[2] != 68: #####
            img = F.interpolate(img, (68, 68))
        return self.model.encode(img).cpu()
    
    def get_feature(self, i, img, centroids):

        img = img[0]
        i = i.item() + 1
        centroids = split_centroids(centroids[0], split = self.split) # all in the third
        img = split_images(img, split = self.split, channel_first = True)
        img = torch.tensor(img).to(self.device)

        return_features = []

        for j in range(img.shape[0]):
            if self.split != 1:
                if len(centroids[j]) < self.min_node: continue
            features = []

            for centroid in torch.tensor(centroids[j]):
                centroid = centroid.to(torch.int32)
                cropped = self.pad_img(centroid, img[j])
                feature = self.get_feature_img(cropped)
                features.append(feature)

            return_features.append(features)

        return return_features
    
    def get_feature_array(self, loader, path, save = True):

        all_features = []
        n_data = len(loader)
        loader = iter(loader)


        # with concurrent.futures.ProcessPoolExecutor(self.num_workers) as executor:
            
        #     # futures = [executor.submit(self.get_feature, i, img, centroids) for i, img, centroids in tqdm(loader)]
        #     chunks = [j * self.chunk_size for j in range(int(n_data / self.chunk_size) + 1)]

        #     # pdb.set_trace()
        #     for start in tqdm(chunks, desc = f'Processing features in chunk size {self.chunk_size}'):
        #         end = min(start + self.chunk_size, n_data)
        #         chunk_features = []
        #         for _ in range(start, end):
        #             j, img, centroids = next(loader)
        #             chunk_features.append(executor.submit(self.get_feature, j, img, centroids))
            
        #         for i, feature in enumerate(concurrent.futures.as_completed(chunk_features)):
        #             all_features.append(feature.result())
        #             chunk_features[i] = 0 # free up storage
            
        for i, img, centroids in tqdm(loader, desc = 'Processing features...'):
            if img is None: continue
            all_features.append(self.get_feature(i, img, centroids))

        if save == True: 
            torch.save(all_features, path)
            return len(all_features)
        else: 
            return all_features
    
class EdgeBuilder:

    def __init__(self, L, n, min_node = 15, split = 4, 
                 theta = 1, num_workers = 4):
        self.L = L
        self.n = n
        self.split = split
        self.min_node = min_node
        self.num_workers = num_workers

    def dist(self, x, y):
        return ((x-y) ** 2).sum()
    
    def get_graph(self, i, centroids):

        centroids = centroids[0] 
        centroids = split_centroids(centroids, split = self.split)
        graphs_ = []

        for centroids_ in centroids:
            if self.split != 1:
                if len(centroids_) < self.min_node: continue

            G = nx.Graph()
            G.add_nodes_from([i + 1 for i in range(centroids_.shape[0])])
            dists, degs = [], [self.n for _ in range(centroids_.shape[0])]
            for j in range(centroids_.shape[0]):
                for k in range(j):
                    dists.append((self.dist(centroids_[j], centroids_[k]), j, k))
            dists = np.array(dists, dtype = [('dist', float), ('i', int), ('j', int)])
            dists.sort(order = 'dist')
            for i in range(dists.shape[0]):
                if dists[i][0] > self.L ** 2: break
                if (degs[dists[i][1]] == 0) | (degs[dists[i][2]] == 0): continue
                degs[dists[i][1]] -= 1
                degs[dists[i][2]] -= 1
                G.add_edge(dists[i][1] + 1, dists[i][2] + 1)
                nx.set_edge_attributes(
                    G, {(dists[i][1] + 1, dists[i][2] + 1): 
                        {'dist': dists[i][0], 'weight': max(0.5,  1.0 - math.sqrt(dists[i][0]) / self.L )}})
                
            graphs_.append(G)
        return graphs_
    
    def get_graphs(self, loader, path, save = True):
        
        graphs = []
        loader = iter(loader)

        # with concurrent.futures.ProcessPoolExecutor(max_workers = self.num_workers) as executor:

            # for i, _, centroids in tqdm(loader, desc = 'Processing graphs'):
            #     futures.append(executor.submit(self.get_graph, i, centroids))
            
            # for i, feature in enumerate(concurrent.futures.as_completed(futures)):
            #     graphs.append(feature.result())

            # futures = [executor.submit(self.get_graph, i, centroids) for i, _, centroids in tqdm(loader, desc = "Processing graphs")]
            # graphs = list(concurrent.futures.Executor.map(lambda x: x.result(), futures))
            # data = [(i, centroids) for i, _, centroids in loader]
            # for graph in executor.map(lambda x: self.get_graph(*x), data):
            #     graphs.append(graph)
            
        for i, img, centroids in tqdm(loader, desc = 'Processing graphs'):
            
            if img is None: continue
            graphs.append(self.get_graph(i, centroids))

            # chunks = [j * self.chunk_size for j in range(int(n_data / self.chunk_size) + 1)]

            # for start in tqdm(chunks, desc = f'Processing graphs in chunk size {self.chunk_size}'):
            #     end = min(start + self.chunk_size, n_data)
            #     chunk_features = []
            #     for _ in range(start, end):
            #         j, _, centroids = next(loader)
            #         chunk_features.append(executor.submit(self.get_graph, j, centroids))
            
            #     for i, feature in enumerate(concurrent.futures.as_completed(chunk_features)):
            #         graphs.append(feature.result())
            #         chunk_features[i] = 0
        
        if save == True:
            with open(path, 'wb') as f:
                pickle.dump(graphs, f)
            return len(graphs)
        else:
            return graphs


class kChannelsGraphBuilder:
    def __init__(self, path, centroids, save = True, save_path = None, normalize = True,
                 channels = 5, img_size = 34, L = 60, n = 5, min_node = 15, theta = 1,
                 split = 4, device = 'cpu', nested_folder = True, paths = None, imgs = None, extract_mode = 'cnn', 
                 load = True, model_path = None, labels = None, illum_which = None, illum = None, 
                 num_workers = 4):
        '''

        Image preprocessing, convert images to a cell-graph with CNN extracted feature for each cell

        Use the CellProfiler Pipeline to obtain the centroids to be passed into this function. Works on any channels images.
        By default, all available workers will be used to speed up computations.

        Args:
        path: path to the image folder. 
        nested_folder: if set to False, we assume there are no nested folders and extract
        path directly from the 'path' directory
        paths: alternatively, you can also provide a list of paths. If provided, 'path' will be ignored.
        imgs: alternatively, you can directly provide the images. If provided, 'path' will be ignored. 
        Expect imgs to be channel-first. Not recommended if you have a large set of data
        centroids: an array of centroids
        save: whether we save the results
        save_path: path to save results
        normalize: whether we remove the brightest pixels before normalization
        img_size: the size of bounding box to use to capture objects
        L: the longest range (in pixels) allowed to build an edge between two nodes
        n: the maximum number of degree a node can have
        theta: controls the impact of neighbors when aggregating node features
        extract_mode: what kind of extractors to use
        load: whether to load path for the feature extractor
        model_path: path to the model pretrained weights
        labels: labels to the images. If provided, it will be saved to the output folder
        illum: optional, set of illumination correction functions to be used on images
        illum_which: optional, a list indicating which illum function should be applied for each image
        num_workers: number of workers. Default: 4

        Returns
        cell features, graph, 
        
        '''
        if nested_folder == False:
            paths = glob(os.path.join(path, '*.tiff'))
        elif type(paths) != type(None):
            if len(paths) != channels:
                paths = group_path_by_channel(paths)
        elif type(imgs) == type(None):
            raise ValueError('Must pass at least one parameter for obtaining data')
        
        self.path = path
        self.channels = channels

        self.datasets = ImageDataset(root_to_img_dir = self.path, normalize = normalize,
                                     centroids = centroids, imgs = imgs, paths = paths, 
                                     illum = illum, illum_which = illum_which)
        
        self.loaders = DataLoader(self.datasets, 
                                  batch_size = 1, sampler=SequentialSampler(self.datasets))
        
        self.extract = FeatureExtractor(img_size=img_size,
            device=device, load = load, split = split, min_node = min_node, 
            load_path = model_path, mode = extract_mode, 
            num_workers = num_workers)
        
        self.edge = EdgeBuilder(L, n, split = split, min_node = min_node, 
                                theta = theta, num_workers = num_workers)

        self.img_size = img_size
        self.save = save
        self.save_path = save_path
        self.labels = labels
        self.split = split
        self.min_node = min_node

    def build(self, extract_edge = True, get_feature = True): 
        returns = []
        if get_feature: 
            extract_returns = self.extract.get_feature_array(
                loader = self.loaders, path = os.path.join(self.save_path, f'nodes.pth'),
                save = self.save)
            if self.save:
                img_loaded = extract_returns
            returns.append(extract_returns)
        if extract_edge: # graph built on the nucleus channel
            edge_returns = self.edge.get_graphs(
                self.loaders, os.path.join(self.save_path, f'graphs_channel.pkl'),
                save = self.save)
            if self.save:
                img_loaded = edge_returns
            returns.append(edge_returns)
        if self.labels is not None:
            new_labels = []
            for i, _, centroids in tqdm(iter(self.loaders)):
                centroids = centroids[0] 
                centroids = split_centroids(centroids, split = self.split)
                for centroids_ in centroids:
                    if len(centroids_) < self.min_node: continue
                    new_labels.append(self.labels[i])
            if self.save:
                torch.save(new_labels, os.path.join(self.save_path, 'labels.pth'))
            img_loaded = len(new_labels)
            returns.append(new_labels)
        
        if self.save:
            return img_loaded
        else:
            return returns
