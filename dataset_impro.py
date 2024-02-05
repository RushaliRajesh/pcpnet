from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
import pdb
from sklearn.cluster import KMeans
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


# do NOT modify the returned points! kdtree uses a reference, not a copy of these points,
# so modifying the points would make the kdtree give incorrect results
def load_shape(point_filename, normals_filename, init_filename, neighs_dists_filename, neighs_inds_filename, curv_filename, pidx_filename):
    pts = np.load(point_filename+'.npy')

    if normals_filename != None:
        normals = np.load(normals_filename+'.npy')
    else:
        normals = None

    if curv_filename != None:
        curvatures = np.load(curv_filename+'.npy')
    else:
        curvatures = None

    if init_filename != None:
        init_normals = np.load(init_filename+'.npy')
    else:
        init_normals = None

    if neighs_dists_filename != None:
        neighs_dists = np.load(neighs_dists_filename+'.npy')
    else:
        neighs_dists = None

    if neighs_inds_filename != None:
        neighs_inds = np.load(neighs_inds_filename+'.npy')
    else:
        neighs_inds = None

    if pidx_filename != None:
        patch_indices = np.load(pidx_filename+'.npy')
    else:
        patch_indices = None

    sys.setrecursionlimit(int(max(1000, round(pts.shape[0]/10)))) # otherwise KDTree construction may run out of recursions
    kdtree = spatial.cKDTree(pts, 10)

    return Shape(pts=pts, kdtree=kdtree, normals=normals, curv=curvatures, pidx=patch_indices, init = init_normals, neighs_dists = neighs_dists, neighs_inds = neighs_inds)

class SequentialPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = None

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, sequential_shapes=True, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))

        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        # return a permutation of the points in the dataset where all points in the same shape are adjacent (for performance reasons):
        # first permute shapes, then concatenate a list of permuted points in each shape
        self.shape_patch_inds = [[]]*len(self.data_source.shape_names)
        point_permutation = []
        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind]+self.data_source.shape_patch_count[shape_ind]

            global_patch_inds = self.rng.choice(range(start, end), size=min(self.patches_per_shape, end-start), replace=False)
            point_permutation.extend(global_patch_inds)

            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():
    def __init__(self, pts, kdtree, normals=None, curv=None, pidx=None, init=None, neighs_dists=None, neighs_inds=None):
        self.pts = pts
        self.kdtree = kdtree
        self.normals = normals
        self.curv = curv
        self.pidx = pidx # patch center points indices (None means all points are potential patch centers)
        self.init = init
        self.neighs_dists = neighs_dists
        self.neighs_inds = neighs_inds


class Cache():
    def __init__(self, capacity, loader, loadfunc):
        # pdb.set_trace()
        self.elements = {}
        self.used_at = {}
        self.capacity = capacity
        self.loader = loader
        self.loadfunc = loadfunc
        self.counter = 0

    def get(self, element_id):
        # pdb.set_trace()
        if element_id not in self.elements:
            # cache miss

            # if at capacity, throw out least recently used item
            if len(self.elements) >= self.capacity:
                remove_id = min(self.used_at, key=self.used_at.get)
                del self.elements[remove_id]
                del self.used_at[remove_id]

            # load element
            self.elements[element_id] = self.loadfunc(self.loader, element_id)

        self.used_at[element_id] = self.counter
        self.counter += 1

        return self.elements[element_id]


class PointcloudPatchDataset(data.Dataset):

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, root, shape_list_filename, patch_radius, points_per_patch, patch_features,
                 seed=None, identical_epochs=False, use_pca=True, center='point', point_tuple=1, cache_capacity=1, point_count_std=0.0, sparse_patches=False):

        # initialize parameters
        
        self.root = root
        self.shape_list_filename = shape_list_filename
        self.patch_features = patch_features
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.identical_epochs = identical_epochs
        self.use_pca = use_pca
        self.sparse_patches = sparse_patches
        self.center = center
        self.point_tuple = point_tuple
        self.point_count_std = point_count_std
        self.seed = seed

        self.include_normals = False
        self.include_curvatures = False
        for pfeat in self.patch_features:
            if pfeat == 'normal':
                self.include_normals = True
            elif pfeat == 'max_curvature' or pfeat == 'min_curvature':
                self.include_curvatures = True
            else:
                raise ValueError('Unknown patch feature: %s' % (pfeat))

        # self.loaded_shape = None
        self.load_iteration = 0
        self.shape_cache = Cache(cache_capacity, self, PointcloudPatchDataset.load_shape_by_index)

        # get all shape names in the dataset
        self.shape_names = []
        with open(os.path.join(root, self.shape_list_filename)) as f:
            self.shape_names = f.readlines()
        self.shape_names = [x.strip() for x in self.shape_names]
        self.shape_names = list(filter(None, self.shape_names))
        

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        # get basic information for each shape in the dataset
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        for shape_ind, shape_name in enumerate(self.shape_names):
            print('getting information for shape %s' % (shape_name))

            # load from text file and save in more efficient numpy format
            point_filename = os.path.join(self.root, shape_name+'.xyz')
            pts = np.loadtxt(point_filename).astype('float32')
            np.save(point_filename+'.npy', pts)

            if self.include_normals:
                normals_filename = os.path.join(self.root, shape_name+'.normals')
                normals = np.loadtxt(normals_filename).astype('float32')
                np.save(normals_filename+'.npy', normals)
            else:
                normals_filename = None

            if self.include_curvatures:
                curv_filename = os.path.join(self.root, shape_name+'.curv')
                curvatures = np.loadtxt(curv_filename).astype('float32')
                np.save(curv_filename+'.npy', curvatures)
            else:
                curv_filename = None

            if self.sparse_patches:
                pidx_filename = os.path.join(self.root, shape_name+'.pidx')
                patch_indices = np.loadtxt(pidx_filename).astype('int')
                np.save(pidx_filename+'.npy', patch_indices)
            else:
                pidx_filename = None

            shape = self.shape_cache.get(shape_ind)

            if shape.pidx is None:
                self.shape_patch_count.append(shape.pts.shape[0])
            else:
                self.shape_patch_count.append(len(shape.pidx))

            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            self.patch_radius_absolute.append([bbdiag * rad for rad in self.patch_radius])

    def process_all(self, labels, i, local_patch, patch_ind, center_point, init_norms):

        labels = torch.from_numpy(labels)
        i = torch.from_numpy(np.array(i))
        local_patch = torch.from_numpy(local_patch)
        center_point = torch.from_numpy(center_point)
        patch_ind = torch.from_numpy(patch_ind)
        norms = torch.from_numpy(init_norms)
        mat = []
        for clus in torch.unique(labels, dim=0):
            ind_clus = torch.where(labels==clus)[0]
            coords = local_patch[ind_clus]

            if(coords.shape[0] < 25):
                for _ in range(25-(coords.shape[0])):
                    a, b = torch.randint(0, coords.shape[0], (2,))
                    # a_n, b_n = torch.divide(a, (a+b)), torch.divide(b, (a+b))
                    a_n = torch.rand(1)
                    b_n = 1-a_n
                    rnd_pt = torch.add(torch.multiply(a_n, coords[a]), torch.multiply(b_n, coords[b])).unsqueeze(0)
                    coords = torch.cat((coords, rnd_pt), dim=0)
            else:
                coords = coords[:25]
            
            vec2 = torch.subtract(center_point, coords)
            closest_ind = torch.argmin(torch.linalg.norm(vec2, dim=1))
            closest = coords[closest_ind]
            vec1 = torch.subtract(center_point, closest)
            vec2 = torch.cat((vec2[:closest_ind], vec2[closest_ind+1:]), dim=0)
            vec1 = torch.divide(vec1, (torch.linalg.norm(vec1)+1e-8))
            vec2 = torch.divide(vec2, (torch.linalg.norm(vec2, dim=1, keepdim=True)+1e-8))
            # vec1 = nn.functional.normalize(vec1, dim=0)
            # vec2 = nn.functional.normalize(vec2, dim=1, p=2, keepdim=True)
            crs_pdts = torch.cross(vec1.expand_as(vec2), vec2)
            area = torch.linalg.norm(crs_pdts, dim=1)
            crs_pdts = torch.divide(crs_pdts, (torch.linalg.norm(crs_pdts, dim=1, keepdim=True)+1e-8))
            # crs_pdts = nn.functional.normalize(crs_pdts, dim=1, p=2, keepdim=True)
            dot_pdts = torch.tensordot(crs_pdts, norms[i].T.to(torch.float32), dims=1)
            # sign = torch.divide(dot_pdts, torch.abs(dot_pdts))
            sign = torch.sign(dot_pdts)
            area = torch.multiply(area, sign)
            sort_indices = torch.argsort(area)
            sorted_coords = coords[sort_indices]
            
            sorted_coords = torch.cat((sorted_coords, torch.sub(center_point, sorted_coords)), dim=1)
            sorted_coords = torch.cat((sorted_coords, norms[i].expand_as(sorted_coords[:,:3])), dim=1)
            # print(sorted_coords.shape)
            
            mat.append(sorted_coords)
            # pdb.set_trace()
        return np.array([np.array(t) for t in mat])
        # return np.array([tensor.cpu().numpy() for tensor in mat])

    def vis(self, pcd, labels, fixed_pnt):
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'pink', 'yellow', 'brown', 'cyan']
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(10):
            cluster_points = pcd[labels == i]
            print(cluster_points.shape)
            print(cluster_points)
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:,2], c=colors[i], label=f'Cluster {i + 1}')

        ax.scatter(fixed_pnt[0], fixed_pnt[1], fixed_pnt[2], c='red', marker='+', s=100, label='Fixed Point')


        ax.set_title('3D Point Clustering based on Radial Distance')
        ax.legend()

        plt.show()

    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        # print("in get item: ", index)
        # pdb.set_trace()
        shape_ind, patch_ind = self.shape_index(index)

        shape = self.shape_cache.get(shape_ind)
        if shape.pidx is None:
            center_point_ind = patch_ind
        else:
            center_point_ind = shape.pidx[patch_ind]

        # get neighboring points (within euclidean distance patch_radius)
        # patch_pts = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3, dtype=torch.float)
        patch_pts = torch.zeros(self.points_per_patch, 3, dtype=torch.float)
        # patch_pts_valid = torch.ByteTensor(self.points_per_patch*len(self.patch_radius_absolute[shape_ind])).zero_()
        patch_pts_valid = []
        scale_ind_range = np.zeros([len(self.patch_radius_absolute[shape_ind]), 2], dtype='int')
        for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
            # pdb.set_trace()
            # patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad))
            # patch_point_dists, patch_point_inds = shape.kdtree.query(shape.pts[center_point_ind, :], k=self.points_per_patch+1)
            neighs_dists = shape.neighs_dists[center_point_ind,1:]
            neighs_inds = shape.neighs_inds[center_point_ind,1:]
            # pdb.set_trace()
            kmeans = KMeans(n_clusters=10).fit(neighs_dists.reshape(-1,1))
            patch_ring_inds = kmeans.labels_
            local_patch = shape.pts[neighs_inds]
            # self.vis(local_patch, patch_ring_inds, shape.pts[center_point_ind])
            mat = self.process_all(patch_ring_inds, center_point_ind, local_patch, neighs_inds, shape.pts[center_point_ind],shape.init)
            
            # start = s*self.points_per_patch
            # end = start+point_count
            # scale_ind_range[s, :] = [start, end]

            # patch_pts_valid += list(range(start, end))

            # convert points to torch tensors
            # patch_pts[start:end, :] = torch.from_numpy(shape.pts[patch_point_inds, :])

            # center patch (central point at origin - but avoid changing padded zeros)
            # if self.center == 'mean':
            #     patch_pts[start:end, :] = patch_pts[start:end, :] - patch_pts[start:end, :].mean(0)
            # elif self.center == 'point':
            #     patch_pts[start:end, :] = patch_pts[start:end, :] - torch.from_numpy(shape.pts[center_point_ind, :])
            # elif self.center == 'none':
            #     pass # no centering
            # else:
            #     raise ValueError('Unknown patch centering option: %s' % (self.center))

            # # normalize size of patch (scale with 1 / patch radius)
            # patch_pts[start:end, :] = patch_pts[start:end, :] / rad


        # if self.include_normals:
        #     patch_normal = torch.from_numpy(shape.normals[center_point_ind, :])

        # if self.include_curvatures:
        #     patch_curv = torch.from_numpy(shape.curv[center_point_ind, :])
        #     # scale curvature to match the scaled vertices (curvature*s matches position/s):
        #     patch_curv = patch_curv * self.patch_radius_absolute[shape_ind][0]

        # if self.use_pca:

        #     # compute pca of points in the patch:
        #     # center the patch around the mean:
        #     pts_mean = patch_pts[patch_pts_valid, :].mean(0)
        #     patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - pts_mean

        #     trans, _, _ = torch.svd(torch.t(patch_pts[patch_pts_valid, :]))
        #     patch_pts[patch_pts_valid, :] = torch.mm(patch_pts[patch_pts_valid, :], trans)

        #     cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
        #     cp_new = torch.matmul(cp_new, trans)

        #     # re-center on original center point
        #     patch_pts[patch_pts_valid, :] = patch_pts[patch_pts_valid, :] - cp_new

        #     if self.include_normals:
        #         patch_normal = torch.matmul(patch_normal, trans)

        # else:
        #     trans = torch.eye(3).float()


        # # get point tuples from the current patch
        # if self.point_tuple > 1:
        #     patch_tuples = torch.zeros(self.points_per_patch*len(self.patch_radius_absolute[shape_ind]), 3*self.point_tuple, dtype=torch.float)
        #     for s, rad in enumerate(self.patch_radius_absolute[shape_ind]):
        #         start = scale_ind_range[s, 0]
        #         end = scale_ind_range[s, 1]
        #         point_count = end - start

        #         tuple_count = point_count**self.point_tuple

        #         # get linear indices of the tuples
        #         if tuple_count > self.points_per_patch:
        #             patch_tuple_inds = self.rng.choice(tuple_count, self.points_per_patch, replace=False)
        #             tuple_count = self.points_per_patch
        #         else:
        #             patch_tuple_inds = np.arange(tuple_count)

        #         # linear tuple index to index for each tuple element
        #         patch_tuple_inds = np.unravel_index(patch_tuple_inds, (point_count,)*self.point_tuple)

        #         for t in range(self.point_tuple):
        #             patch_tuples[start:start+tuple_count, t*3:(t+1)*3] = patch_pts[start+patch_tuple_inds[t], :]


        #     patch_pts = patch_tuples

        # patch_feats = ()
        # for pfeat in self.patch_features:
        #     if pfeat == 'normal':
        #         patch_feats = patch_feats + (patch_normal,)
        #     elif pfeat == 'max_curvature':
        #         patch_feats = patch_feats + (patch_curv[0:1],)
        #     elif pfeat == 'min_curvature':
        #         patch_feats = patch_feats + (patch_curv[1:2],)
        #     else:
        #         raise ValueError('Unknown patch feature: %s' % (pfeat))

        return mat, shape.normals[center_point_ind, :]

    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        # pdb.set_trace()
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind

    # load shape from a given shape index
    def load_shape_by_index(self, shape_ind):
        # pdb.set_trace()
        point_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.xyz')
        normals_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.normals') if self.include_normals else None
        curv_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.curv') if self.include_curvatures else None
        init_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.250norms')
        pidx_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.pidx') if self.sparse_patches else None
        neighs_inds_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.250inds')
        neighs_dists_filename = os.path.join(self.root, self.shape_names[shape_ind]+'.250dists')
        return load_shape(point_filename, normals_filename, init_filename, neighs_dists_filename, neighs_inds_filename, curv_filename , pidx_filename)
