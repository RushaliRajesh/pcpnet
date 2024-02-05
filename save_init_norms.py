import numpy as np 
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_cluster import knn
import open3d as o3d
from scipy.optimize import minimize
from scipy.linalg import lstsq
import torch
import os
import pdb

def pca_plane(points):
    # print(points.shape)
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_[2]/np.linalg.norm(pca.components_[2])

def process_pcd(pcd, pcd_name):
    indices = np.load("pclouds/"+pcd_name+".250inds.npy")
    # _, indices = find_neighs(pcd, 250)
    # print("ind shape:", indices.shape)
    neighs = pcd[indices]
    # print("neighs shape:", neighs.shape)
    norm_esti_pca = Parallel(n_jobs=-1)(delayed(pca_plane)(point_neighs) for point_neighs in neighs)
    return np.array(norm_esti_pca)
    
if __name__ == "__main__":

    shape_names = []

    shape_list_filename = "pclouds/validationset_whitenoise.txt"
    with open(shape_list_filename) as f:
        shape_names = f.readlines()
    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    for shape in shape_names:
        pcd = np.loadtxt("pclouds/"+shape+".xyz") 
        norm_esti_pca = process_pcd(pcd, shape)
        # print("norms shape:", norm_esti_pca.shape)
        file_path = "pclouds/"+shape+".250norms.npy"
        if not os.path.exists(file_path):
            np.save(file_path, norm_esti_pca)
            print(f"Saved norms for {shape}!")
        else:
            print(f"Norms file for {shape} already exists.")
        # np.save("pclouds/"+shape+".250norms", norm_esti_pca)
        # print("saved norms for ", shape, "!")