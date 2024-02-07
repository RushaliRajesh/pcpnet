import os
import sys
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from dataset_impro import PointcloudPatchDataset, SequentialPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler, RandomPointcloudPatchSampler
from model import CNN
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pdb

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./results', help='output folder (estimated point cloud properties)')
    parser.add_argument('--dataset', type=str, default='testset_no_noise.txt', help='shape set file name')
    # parser.add_argument('--dataset', type=str, default='trainingset_whitenoise.txt', help='shape set file name')
    # parser.add_argument('--modeldir', type=str, default='./models', help='model folder')
    # parser.add_argument('--models', type=str, default='single_scale_normal', help='names of trained models, can evaluate multiple models')
    # parser.add_argument('--modelpostfix', type=str, default='_model.pth', help='model file postfix')
    # parser.add_argument('--parmpostfix', type=str, default='_params.pth', help='parameter file postfix')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    parser.add_argument('--sparse_patches', type=int, default=False, help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')
    parser.add_argument('--sampling', type=str, default='full', help='sampling strategy, any of:\n'
                        'full: evaluate all points in the dataset\n'
                        'sequential_shapes_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')
    parser.add_argument('--seed', type=int, default=40938661, help='manual seed')
    parser.add_argument('--batchSize', type=int, default=0, help='batch size, if 0 the training batch size is used')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')

    return parser.parse_args()

def loss_func(pred, target):
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(target, dim=1)
        # dot = torch.abs(torch.diag(torch.dot(pred, target)))
        dot = torch.sum(pred * target, dim=1)
        # loss = torch.mean(torch.acos(dot))   
        loss = 1 - torch.mean(dot)
        # pdb.set_trace()
        return loss

def rms_angular_error(estimated_normals, ground_truth_normals):
    estimated_normals = F.normalize(estimated_normals, dim=1)
    ground_truth_normals = F.normalize(ground_truth_normals, dim=1)

    dot_product = torch.sum(estimated_normals * ground_truth_normals, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angular_diff = torch.acos(dot_product) * torch.div(180.0, torch.pi)
    squared_diff = angular_diff.pow(2)
    mean_squared_diff = torch.mean(squared_diff)
    rms_angular_error = torch.sqrt(mean_squared_diff)

    return rms_angular_error.item()  

def eval(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CNN()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    PATH = "model_modified_cnn.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    pdb.set_trace()

    model_batchSize = 8

    dataset = PointcloudPatchDataset(
        root=opt.indir,
        shape_list_filename=opt.dataset,
        patch_radius=[0.05],
        points_per_patch=1000,
        patch_features=['normal'],
        )
    # datasampler = SequentialPointcloudPatchSampler(dataset)
    datasampler = SequentialShapeRandomPointcloudPatchSampler(
                dataset,
                patches_per_shape=opt.patches_per_shape,
                seed=opt.seed,
                sequential_shapes=True,
                identical_epochs=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=model_batchSize,
        num_workers=int(opt.workers))
    pdb.set_trace()
    shape_ind = 0
    shape_patch_offset = 0
    pred_dim = 0
    pdb.set_trace()

    model_outdir = os.path.join(opt.outdir, "esti_normals")
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    all_loss = []
    all_rms_angular_error = []
    model.eval()
    for ind, (data,norm) in enumerate(dataloader,0):
        data = data.float()
        data = data[:,:,:,:3]
        data = data.permute(0,3,1,2)
        data = data.to(device)
        norm = norm.to(device)
        with torch.no_grad():
            pred = model(data)
        loss = loss_func(pred, norm)
        eval_met = rms_angular_error(pred, norm)
        print(ind)
        print("Loss: ", loss.item())
        print("RMS Angular Error: ", eval_met)
        all_loss.append(loss.item())
        all_rms_angular_error.append(eval_met)
    print("Mean Loss: ", np.mean(all_loss))
    print("Mean RMS Angular Error: ", np.mean(all_rms_angular_error))

        # batch_offset = 0
        # while batch_offset < pred.size(0):

        #     shape_patches_remaining = shape_patch_count-shape_patch_offset
        #     batch_patches_remaining = pred.size(0)-batch_offset

        #     # append estimated patch properties batch to properties for the current shape
        #     shape_properties[shape_patch_offset:shape_patch_offset+min(shape_patches_remaining, batch_patches_remaining), :] = pred[
        #         batch_offset:batch_offset+min(shape_patches_remaining, batch_patches_remaining), :]

        #     batch_offset = batch_offset + min(shape_patches_remaining, batch_patches_remaining)
        #     shape_patch_offset = shape_patch_offset + min(shape_patches_remaining, batch_patches_remaining)

        #     if shape_patches_remaining <= batch_patches_remaining:

        #         normal_prop = shape_properties
        #         np.savetxt(os.path.join(model_outdir, dataset.shape_names[shape_ind]+'.normals'), normal_prop.cpu().numpy())
            
        #         # start new shape
        #         if shape_ind + 1 < len(dataset.shape_names):
        #             shape_patch_offset = 0
        #             shape_ind = shape_ind + 1
                    
        #             shape_patch_count = dataset.shape_patch_count[shape_ind]
        #             # elif opt.sampling == 'sequential_shapes_random_patches':
        #             #     # shape_patch_count = min(opt.patches_per_shape, dataset.shape_patch_count[shape_ind])
        #             #     shape_patch_count = len(datasampler.shape_patch_inds[shape_ind])
        #             # else:
        #             #     raise ValueError('Unknown sampling strategy: %s' % opt.sampling)
        #             shape_properties = shape_properties.new_zeros(shape_patch_count, pred_dim)


        #save the predicted normasl in their respetctive point cloud files, basically segregate and combine the predicted normals according to the point cloud files
if __name__ == '__main__':
    opt = parse_arguments()
    eval(opt)