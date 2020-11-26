'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-26 15:22:36
@LastEditTime: 2019-09-24 11:56:33
@LastEditors: Please set LastEditors
'''
from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np
import visdom
import itertools
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import accuracy, writeOFFfile, get_log, reprocess, smootherror
from models import GCNencoder, GCNdecoder , Discriminator,LSDiscriminator,ZDiscriminator
from PairGraphDataset import GraphDataset, PairGLightColorDataset


# ------------------------------------------------------------------------------------------


path_ae = 'fixedmodel'
check_point = 'checkpoints/gan'
generate_dir = 'generate'
first_tag = True
first_val_tag = True

logger = get_log('train-GANlog.txt')

# Create a directory if not exists
if not os.path.exists(path_ae):
    os.makedirs(path_ae)
if not os.path.exists(generate_dir):
    os.makedirs(generate_dir)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# visdom
viz = visdom.Visdom(env='train-gan')


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=21, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr_G', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--lr_D', type=float, default=0.0004,
                    help='Initial learning rate.')                    
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--z', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--nfeatures', type=int, default=6,
                    help='number of features( 3 features for each node).')
parser.add_argument('--lambda_pair', type=float, default=1,
                    help='weight for mapping loss(A->B or B->A).')                    
parser.add_argument('--datasetA', type=str, default='plane',
                    help='tag of datasetA.')
parser.add_argument('--datasetB', type=str, default='bunny',
                    help='tag of datasetB.')  
parser.add_argument('--unrooledGAN',type=bool,default=True, help='')
parser.add_argument('--d_steps', type=int, default=5,
                    help='iter of training d.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='size of one batch.')
parser.add_argument('--b1',type=float,default=0.5, help='')
parser.add_argument('--b2',type=float,default=0.99, help='')
parser.add_argument('--nnodeA',type=int,default=1000, help='')
parser.add_argument('--nnodeB',type=int,default=1000, help='')
parser.add_argument('--num_pose', type=int, default=6,
                    help='number of random poses for any mesh under an illumination.')
parser.add_argument('--lambda_smooth',type=float,default=2.5, help='')  
parser.add_argument('--lambda_shading',type=float,default=0.3, help='')  
parser.add_argument('--attriA', type=str, default='Light',
                    help='name of attribute of datasetA.(Light,Color)')
parser.add_argument('--attriB', type=str, default='Color',
                    help='name of attribute of datasetB.(Light,Color)') 

parser.add_argument('--path_A_train', type=str, default='Data/light/plane/train',
                    help='the training dataset path for lighting of source domain.')
parser.add_argument('--path_A_val', type=str, default='Data/light/plane/val',
                    help='the val dataset path for lighting of source domain.')
parser.add_argument('--path_B_train', type=str, default='Data/light/bunny/train',
                    help='the training dataset path for lighting of target domain.')
parser.add_argument('--path_B_val', type=str, default='Data/light/bunny/val',
                    help='the val dataset path for lighting of target domain.')
parser.add_argument('--path_color_train', type=str, default='Data/color/bunny/train',
                    help='the training dataset path for color of target domain.')
parser.add_argument('--path_color_val', type=str, default='Data/color/bunny/val',
                    help='the val dataset path for color of target domain.')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if(args.unrooledGAN is False) args.d_steps = 1

# load model
path_A_encoder = os.path.join(path_ae,'{}-encoder.ckpt'.format(args.datasetA))
path_A_decoder = os.path.join(path_ae,'{}-decoder.ckpt'.format(args.datasetA))
path_B_encoder = os.path.join(path_ae,'{}-encoder.ckpt'.format(args.datasetB))
path_B_decoder = os.path.join(path_ae,'{}-decoder.ckpt'.format(args.datasetB))
# path_A2C_generator = os.path.join(path_ae,'{}2{}-Light.ckpt'.format(args.datasetA,args.datasetB))


path_A_train = args.path_A_train
path_A_val = args.path_A_val
path_B_train = args.path_A_train
path_B_val = args.path_B_val
path_color_train = args.path_A_train
path_color_val = args.path_color_val

train_dataset = PairGLightColorDataset(path_A_train,path_B_train,args.datasetA,args.datasetB,path_color_train,args.attriA,args.attriB,args.num_pose)
val_dataset = PairGLightColorDataset(path_A_val,path_B_val,args.datasetA,args.datasetB,path_color_val,args.attriA,args.attriB,args.num_pose)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=3,
                                           drop_last=True,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=args.batch_size,
                                          drop_last=True,
                                          shuffle=True)

# Model and optimizer
encoder_A = GCNencoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnodeA,
                     dropout=args.dropout)
decoder_A = GCNdecoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnodeA,
                     dropout=args.dropout)

encoder_B = GCNencoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnodeB,
                     dropout=args.dropout)
decoder_B = GCNdecoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnodeB,
                     dropout=args.dropout)

encoder_A.load_state_dict(torch.load(path_A_encoder))
decoder_A.load_state_dict(torch.load(path_A_decoder))
encoder_B.load_state_dict(torch.load(path_B_encoder))
decoder_B.load_state_dict(torch.load(path_B_decoder))

for parm in encoder_A.parameters():
    parm.requires_grad = False
for parm in decoder_A.parameters():
    parm.requires_grad = False
for parm in encoder_B.parameters():
    parm.requires_grad = False
for parm in decoder_B.parameters():
    parm.requires_grad = False

encoder_A = nn.DataParallel(encoder_A).to(device)
decoder_A = nn.DataParallel(decoder_A).to(device)
encoder_B = nn.DataParallel(encoder_B).to(device)
decoder_B = nn.DataParallel(decoder_B).to(device)

encoder_A.eval()
decoder_A.eval()
encoder_B.eval()
decoder_B.eval()


Ga2b = nn.Sequential(
        nn.Linear(args.z, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2),
        nn.Linear(512,1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024,2048),
        nn.BatchNorm1d(2048),
        nn.LeakyReLU(0.2),
        nn.Linear(2048,1024),
        nn.BatchNorm1d(1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024,args.z),
        # nn.Tanh()
)

Ga2b = nn.DataParallel(Ga2b).to(device)

Db = LSDiscriminator(nfeat=args.nfeatures,nver=args.nnodeB,dropout=args.dropout)

Db = nn.DataParallel(Db).to(device)


# criterionGAN = torch.nn.BCELoss()
criterionGAN = torch.nn.MSELoss()
criterionPair = torch.nn.L1Loss()
criterionL2 = torch.nn.MSELoss()

optimizer_G = optim.Adam(Ga2b.parameters(),
                       lr=args.lr_G, weight_decay=args.weight_decay,betas=(args.b1,args.b2))
optimizer_D = optim.Adam(Db.parameters(),
                       lr=args.lr_D, weight_decay=args.weight_decay,betas=(args.b1,args.b2))                       

#train
loss_D_B = 0

total_step = len(train_loader)        # A and B has same size

TotalTimeCost=time.time()
for epoch in range(args.epochs):
    CurrEpochTimeCost = time.time()
    for i, (light1, normal1, adj1, light2, normal2, adj2, features_B_color, nonselfadj, fname_A, fname_B) in enumerate(train_loader):
        train_time=time.time()
        Ga2b.train()
        Db.train()
        real_labels = torch.ones(args.batch_size, 1).to(device)
        fake_labels = torch.zeros(args.batch_size, 1).to(device)

        light1 = light1.float().to(device)
        normal1 = normal1.float().to(device)
        adj1 = adj1.float().to(device)
        light2 = light2.float().to(device)
        normal2 = normal2.float().to(device)
        adj2 = adj2.float().to(device)
        features_B_color = features_B_color.float().to(device)
        nonselfadj = nonselfadj.float().to(device)
        
        light1 = light1.view(-1, 1, args.nnodeA, int(args.nfeatures/2))
        normal1 = normal1.view(-1, 1, args.nnodeA, int(args.nfeatures/2))
        adj1 = adj1.view(-1, 1, args.nnodeA, args.nnodeA)
        light2 = light2.view(-1, 1, args.nnodeB, int(args.nfeatures/2))
        normal2 = normal2.view(-1, 1, args.nnodeB, int(args.nfeatures/2))
        adj2 = adj2.view(-1, 1, args.nnodeB, args.nnodeB)
        features_B_color = features_B_color.view(-1, 1,args.nnodeB)
        nonselfadj = nonselfadj.view(-1, 1, args.nnodeB, args.nnodeB)

        #forward ---calculate fake and recons
        # fake_B = decoder_B(Ga2b(encoder_A(input_A,adj_A)),adj_B)
        # loss_G_A2B_pair = criterionPair(fake_B,features_B)*args.lambda_pair
        # fake_B = torch.cat((fake_B,normal256_B),3)
        # fake_B = fake_B.view(-1, 1,args.nnode, args.nfeatures)

        real_B_z = encoder_B(light2, normal2, adj2)
        fake_B_z = Ga2b(encoder_A(light1, normal1, adj1))
        loss_G_A2B_pair = criterionPair(fake_B_z,real_B_z)*args.lambda_pair

        fake_b = decoder_B(fake_B_z, normal2, adj2)
        real_b = decoder_B(real_B_z, normal2, adj2)

        shadingloss = criterionPair(torch.sum(torch.mul(fake_b,normal2),3),features_B_color) * args.lambda_shading
        smoothloss = smootherror(fake_b,nonselfadj) * args.lambda_smooth
        
        fake_b = torch.cat((fake_b,normal2),3)
        fake_b = input_B.view(-1, 1,args.nnodeB, args.nfeatures)
        real_b = torch.cat((real_b,normal2),3)
        real_b = input_B.view(-1, 1,args.nnodeB, args.nfeatures)
        
        #Train D
            
        for k in range(0,args.d_steps):
            pred_B_real = Db(real_b,adj2)
            loss_D_B_real = criterionGAN(pred_B_real,real_labels)

            pred_B_fake = Db(fake_b,adj2)
            loss_D_B_fake = criterionGAN(pred_B_fake,fake_labels)
            loss_D_B_t = (loss_D_B_real + loss_D_B_fake)/2

            optimizer_D.zero_grad()
            loss_D_B_t.backward(retain_graph=True)
            optimizer_D.step()

            if k == 0:
                if(args.unrooledGAN is True):
                    torch.save(Db.state_dict(),os.path.join(check_point,'cacheDb.ckpt'))
                loss_D_B = loss_D_B_t
        
        #Train G

        loss_G_A2B = criterionGAN(Db(fake_b,adj2),real_labels)

        loss_G = loss_G_A2B +loss_G_A2B_pair+ smoothloss+ shadingloss

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if(args.unrooledGAN is True):
            Db.load_state_dict(torch.load(os.path.join(check_point,'cacheDb.ckpt')))
        
        

        logger.info('Epoch [{}/{}], Step [{}/{}], Loss_G: {:.6f}, Loss_G_A2B_pair: {:.6f}, loss_smooth: {:.6f},loss_shading: {:.6f},loss_G_A2B: {:.6f},Loss_D_B: {:.6f}, CurrEpochTime:{:.4f}, TotalTime:{:.4f}'
               .format(epoch+1, args.epochs, i+1, total_step, loss_G.item(),loss_G_A2B_pair.item(),smoothloss.item(),shadingloss.item(),loss_G_A2B.item(),loss_D_B.item(), time.time()-CurrEpochTimeCost,time.time()-TotalTimeCost))
        epoch_iter = (i+1)/total_step
        x = epoch+epoch_iter

        if first_tag == True:
            loss_win = viz.line(Y=np.column_stack((np.array([loss_G.item()]),np.array([loss_G_A2B_pair.item()]),np.array([smoothloss.item()]),np.array([shadingloss.item()]),np.array([loss_G_A2B.item()]),np.array([loss_D_B.item()]))),
                                X=np.column_stack((np.array([x]),np.array([x]),np.array([x]),np.array([x]),np.array([x]),np.array([x]))), opts=dict(legend=['Loss_G','Loss_G_A2B_pair','loss_smooth','loss_shading','loss_G_A2B','Loss_D_B']))
            first_tag = False
        else:
            viz.line(Y=np.column_stack((np.array([loss_G.item()]),np.array([loss_G_A2B_pair.item()]),np.array([smoothloss.item()]),np.array([shadingloss.item()]),np.array([loss_G_A2B.item()]),np.array([loss_D_B.item()]))),
                    X=np.column_stack((np.array([x]),np.array([x]),np.array([x]),np.array([x]),np.array([x]),np.array([x]))), opts=dict(legend=['Loss_G','Loss_G_A2B_pair','loss_smooth','loss_shading','loss_G_A2B','Loss_D_B']),
                    update='append', win=loss_win)

    # test model
    if (epoch+1)%2 == 0:
        Ga2b.eval()
        Db.eval()

        with torch.no_grad():
            loss_A2B = 0
            loss_A2B_RMSE = 0
            loss_min = 1
            loss_max = 0
            test_step = len(val_loader)
            for i, (light1, normal1, adj1, light2, normal2, adj2, features_B_color, nonselfadj, fname_A, fname_B) in enumerate(val_loader):
                
                light1 = light1.float().to(device)
                normal1 = normal1.float().to(device)
                adj1 = adj1.float().to(device)
                light2 = light2.float().to(device)
                normal2 = normal2.float().to(device)
                adj2 = adj2.float().to(device)
                features_B_color = features_B_color.float().to(device)
                nonselfadj = nonselfadj.float().to(device)
                
                light1 = light1.view(-1, 1, args.nnodeA, int(args.nfeatures/2))
                normal1 = normal1.view(-1, 1, args.nnodeA, int(args.nfeatures/2))
                adj1 = adj1.view(-1, 1, args.nnodeA, args.nnodeA)
                light2 = light2.view(-1, 1, args.nnodeB, int(args.nfeatures/2))
                normal2 = normal2.view(-1, 1, args.nnodeB, int(args.nfeatures/2))
                adj2 = adj2.view(-1, 1, args.nnodeB, args.nnodeB)
                features_B_color = features_B_color.view(-1, 1,args.nnodeB)
                nonselfadj = nonselfadj.view(-1, 1, args.nnodeB, args.nnodeB)
                
                
                fake_B = decoder_B(Ga2b(encoder_A(light1, normal1, adj1)), normal2, adj2)

                loss_pair_A2B = criterionPair(fake_B,features_B)
                loss_A2B += loss_pair_A2B
                loss_pair_A2B_L2 = criterionL2(fake_B,features_B)
                loss_A2B_RMSE += math.sqrt(loss_pair_A2B_L2)
                if math.sqrt(loss_pair_A2B_L2) > loss_max:
                    loss_max = math.sqrt(loss_pair_A2B_L2)
                if math.sqrt(loss_pair_A2B_L2) < loss_min:
                    loss_min = math.sqrt(loss_pair_A2B_L2)

                
            logger.info('Test A2B loss of the model on the 1000 test mesh: L1: {}; RMSE:{},min :{},max:{}'.format(
                 loss_A2B / test_step, loss_A2B_RMSE/ test_step,loss_min,loss_max))

            if first_val_tag == True:
                val_win = viz.line(Y=np.column_stack((np.array([loss_A2B.item() / len(test_loader)]))),
                                    X=np.column_stack((np.array([x]))), opts=dict(legend=['val']))
                first_val_tag = False
            else:
                viz.line(Y=np.column_stack((np.array([loss_A2B.item() / len(test_loader)]))),
                        X=np.column_stack((np.array([x]))), opts=dict(legend=['val']),
                        update='append', win=val_win)

        torch.save(Ga2b.module.state_dict(), os.path.join(check_point,'{}2{}-Light-{}.ckpt'.format(args.datasetA,args.datasetB,epoch+1)))
