from __future__ import print_function
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
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial


shape_names = []

shape_list_filename = "pclouds/validationset_whitenoise.txt"
with open(shape_list_filename) as f:
    shape_names = f.readlines()
shape_names = [x.strip() for x in shape_names]
shape_names = list(filter(None, shape_names))

# pdb.set_trace()
# norm_files = [os.path.splitext(os.path.basename(x))[0] for x in norm_files]

def neighs(shape_name):
    # base_pt_name = os.path.splitext(os.path.basename(pt_file))[0]
    # base_norm_name = os.path.splitext(os.path.basename(norm_file))[0]
    #get the ctual paths for pcds from pclouds
    pt_file = "pclouds/"+shape_name+".xyz"

    pcd = np.loadtxt(pt_file)
    
    sys.setrecursionlimit(int(max(1000, round(pcd.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pcd, 10)
    dists, inds = kdtree.query(pcd, k=250)
    # pdb.set_trace()
    dists_file = "pclouds/" + shape_name + ".250dists.npy"
    inds_file = "pclouds/" + shape_name + ".250inds.npy"

    if not os.path.exists(dists_file):
        np.save(dists_file, dists)
        print(f"Saved dists for {shape_name}!")
    else:
        print(f"Dists file for {shape_name} already exists.")

    if not os.path.exists(inds_file):
        np.save(inds_file, inds)
        print(f"Saved inds for {shape_name}!")
    else:
        print(f"Inds file for {shape_name} already exists.")
        
    # np.save("pclouds/"+shape_name+".250dists", dists)
    # np.save("pclouds/"+shape_name+".250inds", inds)
    # print("saved neighs for ", shape_name, "!")

for shape_name in shape_names:
    # CALL function
    neighs(shape_name)
    