'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-09-02 20:12:34
@LastEditTime: 2019-09-22 22:43:12
@LastEditors: Please set LastEditors
'''
from __future__ import division
from __future__ import print_function

import time
import os
import argparse
import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import accuracy, writeColorOFFfile, get_log,load_normal_data,load_adj_data,writeOFFfile,reprocess
from models import GAE, GCNencoder, GCNdecoder,GCNcolorDecoder
from PairGraphDataset import LightColorDataset


# ------------------------------------------------------------------------------------------

# sample_dir = 'plane2bunnyColor'
path_ae = 'fixedmodel'
results = 'results'

first_tag = True

if not os.path.exists(results):
    os.makedirs(results)


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=21, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
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
parser.add_argument('--nnodeA', type=int, default=1000,
                    help='number of class A nodes.')
parser.add_argument('--nnodeB', type=int, default=1000,
                    help='number of class B nodes.')                    
parser.add_argument('--datasetA', type=str, default='plane',
                    help='name of object of datasetA.(plane,bunny)')
parser.add_argument('--datasetB', type=str, default='bunny',
                    help='name of object of datasetB.plane,bunny')
parser.add_argument('--attriA', type=str, default='Light',
                    help='name of attribute of datasetA.(Light,Color)')
parser.add_argument('--attriB', type=str, default='Color',
                    help='name of attribute of datasetB.(Light,Color)')
parser.add_argument('--path_light_test', type=str, default='Data/light/plane/test',
                    help='the testing dataset path for lighting.')
parser.add_argument('--path_color_test', type=str, default='Data/color/bunny/test',
                    help='the testing dataset path for color.')                     

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load model
path_A_encoder = os.path.join(path_ae,'{}-encoder.ckpt'.format(args.datasetA))
path_A_decoder = os.path.join(path_ae,'{}-decoder.ckpt'.format(args.datasetA))
path_B_encoder = os.path.join(path_ae,'{}-encoder.ckpt'.format(args.datasetB))
path_B_decoder = os.path.join(path_ae,'{}-decoder.ckpt'.format(args.datasetB))

path_A2B_generator = os.path.join(path_ae,'{}2{}-Light.ckpt'.format(args.datasetA,args.datasetB))
path_B_light2color_encoder = os.path.join(path_ae,'{}-Light2Color-encoder.ckpt'.format(args.datasetB))
path_B_light2color_decoder = os.path.join(path_ae,'{}-Light2Color-decoder.ckpt'.format(args.datasetB))

path_light_test = args.path_light_test
path_color_test = args.path_color_test
test_dataset = LightColorDataset(path_light_test,path_color_test,args.datasetA,args.datasetB,args.attriA,args.attriB)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
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
encoder_light2color = GCNencoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnodeB,
                     dropout=args.dropout)

decoder_light2color = GCNcolorDecoder(nfeat=args.nfeatures,
                     z=args.z,
                     nver=args.nnodeB,
                     dropout=args.dropout)                     

encoder_A.load_state_dict(torch.load(path_A_encoder))
decoder_A.load_state_dict(torch.load(path_A_decoder))
encoder_B.load_state_dict(torch.load(path_B_encoder))
decoder_B.load_state_dict(torch.load(path_B_decoder))
encoder_light2color.load_state_dict(torch.load(path_B_light2color_encoder))
decoder_light2color.load_state_dict(torch.load(path_B_light2color_decoder))

light2color_model = GAE(encoder_light2color,decoder_light2color)

for parm in encoder_A.parameters():
    parm.requires_grad = False
for parm in decoder_A.parameters():
    parm.requires_grad = False
for parm in encoder_B.parameters():
    parm.requires_grad = False
for parm in decoder_B.parameters():
    parm.requires_grad = False
for parm in light2color_model.parameters():
    parm.requires_grad = False  

encoder_A = nn.DataParallel(encoder_A).to(device)
decoder_A = nn.DataParallel(decoder_A).to(device)
encoder_B = nn.DataParallel(encoder_B).to(device)
decoder_B = nn.DataParallel(decoder_B).to(device)
light2color_model = nn.DataParallel(light2color_model).to(device)

encoder_A.eval()
decoder_A.eval()
encoder_B.eval()
decoder_B.eval()
light2color_model.eval()


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
       #nn.Tanh()
)
Ga2b.load_state_dict(torch.load(path_A2C_generator))
for parm in Ga2b.parameters():
    parm.requires_grad = False
Ga2b = nn.DataParallel(Ga2b).to(device)
Ga2b.eval()

criterion_L1 = torch.nn.L1Loss()
criterion_L2 = torch.nn.MSELoss()



# Test the model
# eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    total_L1 = 0
    total_L2 = 0
    for i, (light1, normal1, adj1, color, normal2, adj2, fname_A, fname_B,nonselfadj) in enumerate(test_loader):

        light1 = light1.float().to(device)
        normal1 = normal1.float().to(device)
        adj1 = adj1.float().to(device)
        color = color.float().to(device)
        normal2 = normal2.float().to(device)
        adj2 = adj2.float().to(device)
        # nonselfadj = nonselfadj.float().to(device)

        light1 = light1.view(-1, 1, args.nnodeA, int(args.nfeatures/2))
        normal1 = normal1.view(-1, 1, args.nnodeA, int(args.nfeatures/2))
        adj1 = adj1.view(-1, 1, args.nnodeA, args.nnodeA)
        color = color.view(-1, 1, args.nnodeB, 1)
        normal2 = normal2.view(-1, 1, args.nnodeB, int(args.nfeatures/2))
        adj2 = adj2.view(-1, 1, args.nnodeB, args.nnodeB)
        # nonselfadj = nonselfadj.view(-1, 1, args.nnodeB, args.nnodeB)

        # Forward pass
        predited_bunny_light = decoder_B(Ga2b(encoder_A(light1, normal1,adj1)),adj2)
        # t = time.time()
        predited_bunny_light = predited_bunny_light.view(-1, 1,args.nnodeB, int(args.nfeatures/2))
        predited_bunny_color = light2color_model(predited_bunny_light,normal2,adj2,normal2,adj2)
        # print(time.time()-t)

        predited_bunny_color = reprocess(predited_bunny_color[0] * 255)
        predited_bunny_color = predited_bunny_color.view(args.nnodeB, 1)
        pointlist = predited_bunny_color.cpu().numpy().tolist()
        
        fname =str(fname_B[0]).replace(args.datasetB,"{}2{}".format(args.datasetA,args.datasetB))
        writeColorOFFfile(os.path.join(
            results, fname), pointlist, os.path.join(args.path_color_test,fname_B))


        loss_L1 = criterion_L1(predited_bunny_color, color)
        total_L1 += loss_L1

        loss_L2 = criterion_L2(predited_bunny_color, color)
        total_L2 += math.sqrt(loss_L2)
        # print('L1:{}-{}'.format(fname_B,loss_L1))
    

    print('Test loss of the model on the {} test mesh: L1: {};RMSE: {}'.format(
        len(test_loader),total_L1 / len(test_loader),total_L2/len(test_loader)))

