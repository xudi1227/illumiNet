'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-23 22:07:37
@LastEditTime: 2019-09-23 21:45:15
@LastEditors: Please set LastEditors
'''
from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import accuracy, writeColorOFFfile, get_log, loadnormal,load_adj_data,load_nonself_adj,smootherror
from models import GAE, GCNencoder,GCNcolorDecoder
import visdom
from PairGraphDataset import LightColorDataset


# ------------------------------------------------------------------------------------------

sample_dir = 'color'
check_point = 'check_points'
first_train_tag = True
first_val_tag = True

logger = get_log('light2color-log.txt')

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(check_point):
    os.makedirs(check_point)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=21, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--batch_size', type=int, default=256,
                    help='the size of a batch .')
parser.add_argument('--z', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nfeatures', type=int, default=6,
                    help='number of features( 3 features for each node).')
parser.add_argument('--nnode', type=int, default=1000,
                    help='number of nodes.')
parser.add_argument('--name_mesh', type=str, default='bunny',
                    help='name of current target mesh domain.') 
parser.add_argument('--attriA', type=str, default='Light',
                    help='name of attribute of datasetA.(Light,Color)')
parser.add_argument('--attriB', type=str, default='Color',
                    help='name of attribute of datasetB.(Light,Color)')
parser.add_argument('--path_light_train', type=str, default='Data/light/bunny/train',
                    help='the training dataset path for lighting.')
parser.add_argument('--path_color_train', type=str, default='Data/color/bunny/train',
                    help='the training dataset path for color.')
parser.add_argument('--path_light_val', type=str, default='Data/light/bunny/val',
                    help='the val dataset path for lighting.')
parser.add_argument('--path_color_val', type=str, default='Data/color/bunny/val',
                    help='the val dataset path for color.')
parser.add_argument('--lambda_smooth', type=float, default=0.05,
                    help='the weight of smooth loss.')
                                        
                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


path_light_train = args.path_light_train
path_color_train = args.path_color_train
path_light_val = args.path_light_val
path_color_val = args.path_color_val

train_dataset = LightColorDataset(path_light_train,path_color_train,args.name_mesh,args.name_mesh,args.attriA,args.attriB)
val_dataset = LightColorDataset(path_light_val,path_color_val,args.name_mesh,args.name_mesh,args.attriA,args.attriB)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=5,
                                           drop_last=True,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=True)

viz = visdom.Visdom(env='train-gae')

# Model and optimizer
encoder = GCNencoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnode,
                     dropout=args.dropout)

decoder = GCNcolorDecoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnode,
                     dropout=args.dropout)

model = GAE(encoder, decoder)
# multi-GPUs
model = nn.DataParallel(model)
model.to(device)

criterion_train = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# Train the model
total_step = len(train_loader)
TotalTimeCost=time.time()
test_iter = iter(val_loader)
for epoch in range(args.epochs):
    CurrEpochTimeCost = time.time()
    for i, (light, normal, adj, color, fname_A, fname_B, nonselfadj) in enumerate(train_loader):
        train_time=time.time()
        model.train()
        
        
        light = light.float().to(device)
        normal = normal.float().to(device)
        adj = adj.float().to(device)
        color = color.float().to(device)
        nonselfadj = nonselfadj.float().to(device)

        light = light.view(-1, 1, args.nnode, int(args.nfeatures/2))
        normal = normal.view(-1, 1, args.nnode, int(args.nfeatures/2))
        adj = adj.view(-1, 1, args.nnode, args.nnode)
        color = color.view(-1, 1, args.nnode, 1)
        nonselfadj = nonselfadj.view(-1, 1, args.nnode, args.nnode)

        # Forward pass
        
        predicted_color = model(light, normal, adj, normal, adj)
        smoothloss = args.lambda_smooth * smootherror(predicted_color,nonselfadj)
        loss = criterion(predicted_color, color)+smoothloss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print('Train time each batch:{:.4f}'.format(time.time()-train_time))
        logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f},smoothloss:{:.6f}, CurrEpochTime:{:.4f}, TotalTime:{:.4f}'
               .format(epoch+1, args.epochs, i+1, total_step, loss.item(), smoothloss.item(),time.time()-CurrEpochTimeCost,time.time()-TotalTimeCost))
        epoch_iter = (i+1)/total_step
        x = epoch+epoch_iter

        if first_train_tag == True:
            loss_win = viz.line(Y=np.column_stack((np.array([loss.item()]),np.array([smoothloss.item()]))),
                                X=np.column_stack((np.array([x]),np.array([x]))), opts=dict(legend=['loss','smoothloss']))
            first_train_tag = False
        else:
            viz.line(Y=np.column_stack((np.array([loss.item()]),np.array([smoothloss.item()]))),
                    X=np.column_stack((np.array([x]),np.array([x]))), opts=dict(legend=['loss','smoothloss']),
                    update='append', win=loss_win)
    
    
    if (epoch+1) % 10 == 0:
        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            total = 0
            loss_min = 1
            loss_max = 0
            for i, (light, normal, adj, color, fname_A, fname_B, nonselfadj) in enumerate(val_loader):

                light = light.float().to(device)
                normal = normal.float().to(device)
                adj = adj.float().to(device)
                color = color.float().to(device)
                nonselfadj = nonselfadj.float().to(device)

                light = light.view(-1, 1, args.nnode, int(args.nfeatures/2))
                normal = normal.view(-1, 1, args.nnode, int(args.nfeatures/2))
                adj = adj.view(-1, 1, args.nnode, args.nnode)
                color = color.view(-1, 1, args.nnode, 1)
                nonselfadj = nonselfadj.view(-1, 1, args.nnode, args.nnode)
            
                # Forward pass
                predicted_color = model(light, normal, adj, normal, adj)
                loss = criterion(predicted_color, color)
                total += loss
                if loss > loss_max:
                    loss_max = loss
                if loss < loss_min:
                    loss_min = loss

                # if i == 0:
                #     # recons = recons[0] * 127.5+127.5
                #     predicted_color = predicted_color[0] * 255
                #     predicted_color = predicted_color.view(args.nnode, 1)
                #     pointlist = predicted_color.cpu().numpy().tolist()
                #     writeColorOFFfile(os.path.join(
                #         sample_dir, 'epoch{}_{}'.format(epoch+1,str(fname_B[0]))), pointlist, os.path.join(
                #         path_light_test, '{}'.format(str(fname_A[0]))))

            logger.info('Test loss of the model on the 1000 val mesh: {},min:{},max:{}'.format(
                total / len(val_loader),loss_min,loss_max))

            if first_val_tag == True:
                val_win = viz.line(Y=np.column_stack((np.array([total.item() / len(val_loader)]))),
                                    X=np.column_stack((np.array([x]))), opts=dict(legend=['val']))
                first_val_tag = False
            else:
                viz.line(Y=np.column_stack((np.array([total.item() / len(val_loader)]))),
                        X=np.column_stack((np.array([x]))), opts=dict(legend=['val']),
                        update='append', win=val_win)
        # Save the model checkpoint
        # torch.save(model.state_dict(), os.path.join(check_point,'model-{}.ckpt'.format(epoch+1)))
        torch.save(model.module.encoder.state_dict(), os.path.join(check_point,'{}-{}2{}-encoder-{}.ckpt'.format(args.name_mesh,args.attriA,args.attriB,epoch+1)))
        torch.save(model.module.decoder.state_dict(), os.path.join(check_point,'{}-{}2{}-decoder-{}.ckpt'.format(args.name_mesh,args.attriA,args.attriB,epoch+1)))
