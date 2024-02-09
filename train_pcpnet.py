from __future__ import print_function

import argparse
import os
import sys
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import pdb
from model import CNN
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# from tensorboardX import SummaryWriter # https://github.com/lanpa/tensorboard-pytorch
# from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
# from dataset_impro import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from dataset_diff import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--name', type=str, default='my_single_scale_normal', help='training run name')
    parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.', help='description')
    parser.add_argument('--indir', type=str, default='./pclouds', help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='./models', help='output folder (trained models)')
    parser.add_argument('--logdir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str, default='trainingset_whitenoise.txt', help='training set file name')
    parser.add_argument('--testset', type=str, default='validationset_whitenoise.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int, default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='', help='refine model at this path')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0, help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=1000, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=100, help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False, help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--normal_loss', type=str, default='ms_euclidean', help='Normal loss type:\n'
                        'ms_euclidean: mean square euclidean distance\n'
                        'ms_oneminuscos: mean square 1-cos(angle error)')

    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['unoriented_normals'], help='outputs of the network, a list with elements of:\n'
                        'unoriented_normals: unoriented (flip-invariant) point normals\n'
                        'oriented_normals: oriented point normals\n'
                        'max_curvature: maximum curvature\n'
                        'min_curvature: mininum curvature')
    parser.add_argument('--use_point_stn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max', help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1, help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int, default=500, help='max. number of points per patch')

    return parser.parse_args()

def cos_angle(v1, v2):
    v1 = nn.functional.normalize(v1, dim=1)
    v2 = nn.functional.normalize(v2, dim=1)
    # pdb.set_trace()
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)
    
def loss_func(pred, target):
    loss = (1-cos_angle(pred, target)).pow(2).mean() 
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

def train_pcpnet(opt):

    device = torch.device("cpu" if opt.gpu_idx < 0 else "cuda:%d" % opt.gpu_idx)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))

    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (opt.name))
        if response == 'y':
            if os.path.exists(log_dirname):
                shutil.rmtree(os.path.join(opt.logdir, opt.name))
        else:
            sys.exit()

    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    output_loss_weight = []
    pred_dim = 0
    for o in opt.outputs:
        if o == 'unoriented_normals' or o == 'oriented_normals':
            if 'normal' not in target_features:
                target_features.append('normal')

            output_target_ind.append(target_features.index('normal'))
            output_pred_ind.append(pred_dim)
            output_loss_weight.append(1.0)
            pred_dim += 3
        elif o == 'max_curvature' or o == 'min_curvature':
            if o not in target_features:
                target_features.append(o)

            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            if o == 'max_curvature':
                output_loss_weight.append(0.7)
            else:
                output_loss_weight.append(0.3)
            pred_dim += 1
        else:
            raise ValueError('Unknown output: %s' % (o))

    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')

    # # create model
    # if len(opt.patch_radius) == 1:
    #     pcpnet = PCPNet(
    #         num_points=opt.points_per_patch,
    #         output_dim=pred_dim,
    #         use_point_stn=opt.use_point_stn,
    #         use_feat_stn=opt.use_feat_stn,
    #         sym_op=opt.sym_op,
    #         point_tuple=opt.point_tuple)
    # else:
    #     pcpnet = MSPCPNet(
    #         num_scales=len(opt.patch_radius),
    #         num_points=opt.points_per_patch,
    #         output_dim=pred_dim,
    #         use_point_stn=opt.use_point_stn,
    #         use_feat_stn=opt.use_feat_stn,
    #         sym_op=opt.sym_op,
    #         point_tuple=opt.point_tuple)

    # if opt.refine != '':
    #     pcpnet.load_state_dict(torch.load(opt.refine))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)

    print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    import time
    
    # create train and test dataset loaders
    # pdb.set_trace()
    train_dataset = PointcloudPatchDataset(
        # task = 'direct_pos_3_dim',
        root=opt.indir,
        shape_list_filename=opt.trainset,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    
    # def encode_position(x, enc_dim):
        
    #     positions = [x]

    #     for i in range(enc_dim):
    #         print(torch.sin( (2.0**i) *  x).shape)
    #         positions.append(torch.sin( (2.0**i) *  x ))
    #         positions.append(torch.cos( (2.0**i) *  x ))
        
    #     print(len(positions))
    #     return torch.concat(positions, dim=2)
    
    # x = train_dataset[34]
    # print(x[0].shape)
    # print(encode_position(torch.tensor(x[0][:,:,:3]), 10).shape)

    x = train_dataset[284492]
    print(x)
    
    
    # print(len(train_dataset))
    # # t = time.time()
    # # kuchbhi = train_dataset[7]

    # # pdb.set_trace()
    # # for i in range(len(train_dataset)):
    # #     print(train_dataset[i][0].shape)
    # #     print(train_dataset[i][1].shape)
        
        
    # # out = train_dataset[222]
    # # print("c: ",c)
    # # print(time.time()-t)    
    
    # # if opt.training_order == 'random':
    # train_datasampler = RandomPointcloudPatchSampler(
    #     train_dataset,
    #     patches_per_shape=opt.patches_per_shape,
    #     seed=opt.seed,
    #     identical_epochs=opt.identical_epochs)
    
    # # c=0
    # # for i in train_dataset:
    # #     c=c+1
    # #     print(train_dataset[0])
    # # print(c)        

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     sampler=train_datasampler,
    #     batch_size=opt.batchSize,
    #     num_workers=int(opt.workers))
    
    # for i in train_dataloader:
    #     print(len(i[0]))
    #     break
    

    # test_dataset = PointcloudPatchDataset(
    #     task = 'direct_pos_3_dim',
    #     root=opt.indir,
    #     shape_list_filename=opt.testset,
    #     patch_radius=opt.patch_radius,
    #     points_per_patch=opt.points_per_patch,
    #     patch_features=target_features,
    #     point_count_std=opt.patch_point_count_std,
    #     seed=opt.seed,
    #     identical_epochs=opt.identical_epochs,
    #     use_pca=opt.use_pca, 
    #     center=opt.patch_center,
    #     point_tuple=opt.point_tuple,
    #     cache_capacity=opt.cache_capacity)
    # y = test_dataset[99]
    # print(y)
    # test_datasampler = RandomPointcloudPatchSampler(
    #     test_dataset,
    #     patches_per_shape=opt.patches_per_shape,
    #     seed=opt.seed,
    #     identical_epochs=opt.identical_epochs)

    # test_dataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     sampler=test_datasampler,
    #     batch_size=opt.batchSize,
    #     num_workers=int(opt.workers))

    # pdb.set_trace()

    # print('training set: %d patches (in %d batches) ' %
    #       (len(train_datasampler), len(train_dataloader)))

#     try:
#         os.makedirs(opt.outdir)
#     except OSError:
#         pass
    
    
#     model = CNN()
#     device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#     print("device: ", device)
#     model = model.to(device)
#     # if torch.cuda.device_count() > 1:
#     #     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     #     model = nn.DataParallel(model)

#     # model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,20,30], gamma=0.1) # milestones in number of optimizer iterations
#     tr_loss_per_epoch = []
#     val_loss_per_epoch = []

#     # checkpoint = torch.load("model_prev_cnn_with_val.pt")
#     # model.load_state_dict(checkpoint['model_state_dict'])
#     # hehe_epoch = checkpoint['epoch']

#     for epoch in range(opt.nepoch):
#         train_loss = []
#         val_loss = []
#         eval_tr_ls = []
#         eval_val_ls = []
#         model.train()
#         for i, (data, norms) in enumerate(train_dataloader, 0):
#             inputs = data.float()
#             inputs = inputs[:,:,:,:3]
#             # pdb.set_trace()
#             inputs = inputs.permute(0,3,1,2)
#             # pdb.set_trace()
#             norms = norms.float()
#             inputs, norms = inputs.to(device), norms.to(device)
            
#             out = model(inputs)
            
#             loss = loss_func(out, norms)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss.append(loss.item())
#             eval_tr_ls.append(rms_angular_error(out, norms))
#             # pdb.set_trace()

#         bef_lr = optimizer.param_groups[0]['lr']
#         scheduler.step()
#         aft_lr = optimizer.param_groups[0]['lr']
#         print(f'epoch: {epoch}, learning rate: {bef_lr} -> {aft_lr}')

#         tot_train_loss = np.mean(train_loss)  
#         tr_loss_per_epoch.append(tot_train_loss)  
#         eval_tr = np.mean(eval_tr_ls)
#         print(f'epoch: {epoch}, training loss: {tot_train_loss}, training RMS Angular Error :P : {eval_tr}')
        
#         with torch.no_grad():
#             model.eval()
#             for i, (data, norms) in enumerate(test_dataloader, 0):
#                 inputs = data.float()
#                 inputs = inputs[:,:,:,:3]
#                 inputs = inputs.permute(0,3,1,2)
#                 norms = norms.float()
#                 inputs, norms = inputs.to(device), norms.to(device)
                
#                 out = model(inputs)
#                 loss = loss_func(out, norms)

#                 val_loss.append(loss.item())
#                 eval_val_ls.append(rms_angular_error(out, norms))
                
#         tot_val_loss = np.mean(val_loss)
#         val_loss_per_epoch.append(tot_val_loss)
#         eval_val = np.mean(eval_val_ls)
#         print(f'epoch: {epoch}, val loss: {tot_val_loss}, val RMS Angular Error :P : {eval_val}')

#         if epoch % 10 == 0:
#             # Additional information
#             EPOCH = epoch
#             PATH = "model_prev_cnn_with_val.pt"
#             LOSS = tot_train_loss

#             torch.save({
#                         'epoch': EPOCH,
#                         'model_state_dict': model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'loss': LOSS,
#                         'batchsize' : opt.batchSize,
#                         'val_losses_so_far' : val_loss_per_epoch,
#                         'train_losses_so_far' : tr_loss_per_epoch
#                         }, PATH)
#             print("Model saved at epoch: ", epoch)
            


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)