import os
import logging
import argparse
from glob import glob
import pickle

import numpy as np
import pandas as pd
import scipy.io
import torch
from tqdm import tqdm

from CSIP.build_graph import GraphBuilder, kChannelsGraphBuilder
from CSIP.utils import unload_data, reduce_bounding_boxes_on_images, group_path_by_channel

import pdb

def retrieve_pos(path):
    path = os.path.basename(path)
    parts = path.split("_")
    return (ord(parts[1][0]) - ord('a') + 1, int(parts[1][1:]), int(parts[2][1]))

def retrieve_idx(row, col, site):
    return 6 * (24 * (row - 1) + col - 1) + site - 1

def rev_retrieve_idx(idx):
    idx-=1
    return (int(idx/6/24) + 1, int(idx/6) % 24 + 1, idx %6+1)

def process_data(args):

    logging.getLogger().setLevel(logging.INFO)

    all_centroids, broad_id, plates = [], [], []
    all_plates = ['24639'] #sorted(next(os.walk(args.metadata_path))[1]) 
    logging.info(f"{len(all_plates)} detected: {','.join(all_plates)}")
    suffices_imgs = ['-Hoechst', '-ERSytoBleed', '-ERSyto', '-Ph_golgi', '-Mito']

    for plate in tqdm(all_plates, desc = 'Loading metadata...'): # just pick first 10

        try:

            for suf in suffices_imgs:
                if os.path.exists(os.path.join(args.data_path, plate + suf)) == False:
                    raise FileNotFoundError
                
            centroid_path = os.path.join(args.metadata_path, plate, args.centroids_file)
            metadata_path = os.path.join(args.metadata_path, plate, args.metadata_file)

            if (os.path.exists(centroid_path) & os.path.exists(metadata_path)) == False:
                raise FileNotFoundError
            
            centroids = pd.read_csv(centroid_path)
            all_centroids.append(centroids)
            metadata = pd.read_csv(metadata_path)
            metadata['row'] = metadata.Metadata_Well.apply(lambda x: ord(x[0]) - ord('a'))
            metadata['col'] = metadata.Metadata_Well.apply(lambda x: int(x[1:]) - 1)
            broad_id.append(metadata.loc[:, ['row', 'col', 'Metadata_broad_sample']])
            plates.append(plate)

        except FileNotFoundError:
            logging.warn(f"Missing either cell coordinate file, cell image, or metadata for plate {plate}, skipping the plate")
            

    # create an index of files to drop due to bad quality/ lack of information
    drop_idxs_by_plate = [[] for _ in range(len(plates))]
    logging.info("Filtering out poor-quality images...")
    for i, centroids in enumerate(all_centroids):
        if centroids.ImageNumber.nunique()!=2304:
            for j in range(1,2305):
                if j not in centroids.ImageNumber:
                    drop_idxs_by_plate[i].append(j)
        quality_control = pd.read_csv(os.path.join(args.metadata_path, plates[i], args.quality_control))
        quality_control['row'] = quality_control.Image_Metadata_Well.apply(lambda x: ord(x[0]) - ord('a'))
        quality_control['col'] = quality_control.Image_Metadata_Well.apply(lambda x: int(x[1:]) - 1)
        quality_control['ImageNumber'] = 6 * (24 * quality_control.row + quality_control.col) + quality_control.Image_Metadata_Site
        bad_pos = quality_control.loc[
            (quality_control.Image_Metadata_isBlurry == 1) | 
            (quality_control.Image_Metadata_isSaturated == 1 ), "ImageNumber"].values.tolist()
        drop_idxs_by_plate[i].extend(bad_pos)

    centroids_by_img, pert_name, pert_smiles = [], [], []
    img_idxs_by_plate = []

    # You may have to rewrite the functions to retrieve SMILES, 
    # compound name, and other relevant information for each image 
    # if your data is stored differently.

    logging.info("Adding metadata...")
    for i, (centroids, metadata, drops) in enumerate(
        zip(all_centroids, broad_id, drop_idxs_by_plate)):
        
        idxs_to_use = []
        for j in centroids.ImageNumber.unique(): 
            if j in drops: continue
            idxs_to_use.append(j)
            centroids_by_img.append(centroids.loc[
                centroids.ImageNumber == j, [
                    "AreaShape_Center_X", "AreaShape_Center_Y"]].values)
            
        img_idxs_by_plate.append(idxs_to_use)
        rows, cols, _ = zip(*[rev_retrieve_idx(j) for j in idxs_to_use])
        id_to_name = pd.read_csv(args.conversion)
        for j in range(len(idxs_to_use)):
            try: 
                sample_id = metadata.loc[(metadata.row == rows[j]) & (metadata.col == cols[j]), "Metadata_broad_sample"].values[0]
                name = id_to_name.loc[id_to_name.BROAD_ID == sample_id, "CPD_NAME"].values[0]
                smile = id_to_name.loc[id_to_name.BROAD_ID == sample_id, "CPD_SMILES"].values[0]    
            except IndexError: # control group
                name = 'DMSO'
                smile = 'CS(=O)C'
            pert_name.append(name)
            pert_smiles.append(smile)
    
    with open(os.path.join(args.save, "smiles.pkl"), "wb") as fp:
        pickle.dump(pert_smiles, fp)

    with open(os.path.join(args.save, "pert_names.pkl"), "wb") as fp:
        pickle.dump(pert_name, fp)

    paths = [[], [], [], [], []]
    if args.illum_path is not None:
        illum, illum_which = [], []

    # might need to be changed based on dataset
    channels = ['ERSyto', 'ERSytoBleed', 'Hoechst', 'Mito', 'Ph_golgi']
    logging.info("Collecting paths...")
    for i, plate in enumerate(plates):
        for (j, channel) in enumerate(channels):
            channel_path = os.path.join(args.data_path, plate[-5:] + "-" + channel, "*.tif")
            all_paths = glob(channel_path)
            pos_to_path = {}
            for k in range(len(all_paths)):
                pos_to_path[retrieve_pos(all_paths[k])] = all_paths[k]

            for k in img_idxs_by_plate[i]:
                paths[j].append(pos_to_path[rev_retrieve_idx(k)])

        if args.illum_path is not None:
            illum_ = []
            for j in range(len(channels)):
                illum_path_split =  args.illum_path[j].split('(plate)')
                if len(illum_path_split) == 1: 
                    cur_illum_path = args.illum_path[j]
                else:
                    cur_illum_path = ''
                    for k, path in enumerate(illum_path_split): 
                        cur_illum_path += path
                        if k != len(illum_path_split) - 1:
                            cur_illum_path += plates[i][-5:]

                illum_.append(scipy.io.loadmat(
                    os.path.join(args.metadata_path, plates[i], cur_illum_path))['Image']
                )
            illum.append(torch.tensor(illum_, dtype = torch.float32))
            for j in img_idxs_by_plate[i]:
                illum_which.append(len(illum) - 1)

    logging.info("Reducing redundant bounding boxes...")
    centroids_by_img = reduce_bounding_boxes_on_images(centroids_by_img, args.img_size, args.bbox_max_overlap)

    logging.info("Building graphs...")
    builder = kChannelsGraphBuilder(
        path = args.data_path, paths = paths, centroids = centroids_by_img, save = True,
        save_path = args.save, n = args.n, L = args.L, theta = args.theta, device = args.device, channels = len(channels),
        extract_mode = args.extract_mode, load = args.load, split = args.split, min_node = args.min_node, 
        model_path = args.model_path, img_size = args.img_size, illum = illum, illum_which = illum_which,
        chunk_size = args.chunk_size, num_workers = args.num_workers)
    img_loaded = builder.build(get_feature = args.extract_feature, extract_edge = args.extract_edge)
    
    logging.info(f"All processes finished. Number of image loaded: {img_loaded}") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str, default = None, help = "Path to dataset, which should be organized by plate subfolders")
    parser.add_argument("--metadata_path", type = str, default = None, help = "Path to metadata")
    parser.add_argument("--save", type = str, default = "output", help = "Where we save the processed files")
    parser.add_argument("--centroids_file", type = str, default = None, help = "Name of the centroids csv files, which should be stored in the metadata folder organized by plate subfolders")
    parser.add_argument("--metadata_file", type = str, default = None, help = "the name of the metadata csv file in each plate subfolder")
    parser.add_argument("--quality_control", type = str, default = None, help = "optional, quality control to filter out low-quality images")
    parser.add_argument("--conversion", type = str, default = None, help = "optional. If Broad ID is used to identify compound, a table can be used for conversion")
    parser.add_argument("--img_size", type = int, default = 25, help = "the size of bounding box")
    parser.add_argument("--bbox_max_overlap", type = float, default = 0.4, help = "Delete one of two bounding boxes with more than this percentage of overlapping")
    parser.add_argument("--L", type = int, default = 150, help = 'maximum distance between two objects to construct edge')
    parser.add_argument("--n", type = int, default = 7, help = 'maximum neighbors a node can have')
    parser.add_argument("--theta", type = int, default = 1, help = 'regulate the impact of neighbor features')
    parser.add_argument("--split", type = int, default = 1, help = 'whether we split images. Can be either 1 or 4.')
    parser.add_argument("--min_node", type = int, default = 12, help = 'minimum number of nodes for each image')
    parser.add_argument("--chunk_size", type = int, default = 2000, help = "the maximum number of data we process at the same time.")
    parser.add_argument("--imgs_per_site", type = int, default = 6, help = 'number of images per site in the original plate')
    parser.add_argument("--device", type = str, default = "cpu", help = '')
    parser.add_argument("--extract_mode", type = str, default = "cnn", help = 'can be "cnn" or "interpolate" ')
    parser.add_argument("--model_path", type = str, default = None, help = 'if cnn is used for feature extraction, you can provide a model path')
    parser.add_argument("--load", type = int, default = 0, help = 'if we load model from path, default: 0')
    parser.add_argument("--illum_path", nargs = '+', type = str, default = None, help = "optional, list of names of illumination correction function, which should be stored identically in each plate subfolder")
    parser.add_argument("--num_workers", type = int, default = None, help = "")
    parser.add_argument("--extract_feature", type = int, default = 1, help = "whether we extract features, default: 1")
    parser.add_argument("--extract_edge", type = int, default = 1, help = "whether we extract edges, default: 1")
    args = parser.parse_args() 
    
    process_data(args)

if __name__ == "__main__":
    main()
