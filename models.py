'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-19 20:11:09
@LastEditTime: 2019-09-06 15:27:05
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,GConv


class GCNencoder(nn.Module):
    def __init__(self, nfeat, z, dropout, nver):
        super(GCNencoder, self).__init__()

        self.nver = nver
        self.nfeat = nfeat
        self.gc1 = GraphConvolution(nfeat, 3*nfeat)
        self.gc2 = GraphConvolution(3*nfeat, int(4*nfeat))
        self.bn1 = nn.BatchNorm2d(1)
        self.fc = nn.Linear(int(4*nver*nfeat), z)
        self.dropout = dropout

    def forward(self, l, n, adj):
        out = torch.cat((l,n),3)
        out = torch.tanh(self.bn1(self.gc1(out, adj)))
        out = F.dropout(out, self.dropout, training=self.training)
        
        out = torch.tanh(self.gc2(out, adj))
        out = F.dropout(out, self.dropout, training=self.training)
        out = out.view(-1, int(4*self.nver*self.nfeat))
        out = self.fc(out)
        return out


class GCNdecoder(nn.Module):
    def __init__(self, nfeat, z, dropout, nver):
        super(GCNdecoder, self).__init__()

        self.nver = nver
        self.nfeat = nfeat
        self.fc = nn.Linear(z, int(4*nver*nfeat))
        self.bnfc = nn.BatchNorm1d(int(4*nver*nfeat))
        self.gc1 = GraphConvolution(int(4*nfeat+3), 3*nfeat)
        self.bn1 = nn.BatchNorm2d(1)
        self.gc2 = GraphConvolution(3*nfeat, int(nfeat/2))

        self.dropout = dropout

    def forward(self, z , n, adj):

        out = self.fc(z)
        out = torch.tanh(self.bnfc(out))
        out = out.view(-1, 1, self.nver, int(4*self.nfeat))
        out = torch.cat((out,n),3)
        out = torch.tanh(self.bn1(self.gc1(out, adj)))
        out = F.dropout(out, self.dropout, training=self.training)
        out = torch.tanh(self.gc2(out, adj))

        return out

class GCNcolorDecoder(nn.Module):
    def __init__(self, nfeat, z, dropout, nver):
        super(GCNcolorDecoder, self).__init__()

        self.nver = nver
        self.nfeat = nfeat
        self.fc = nn.Linear(z, int(4*nver*nfeat))
        self.bnfc = nn.BatchNorm1d(int(4*nver*nfeat))
        self.gc1 = GraphConvolution(int(4*nver*nfeat)+3, int(3*nfeat))
        self.bn1 = nn.BatchNorm2d(1)
        self.gc2 = GraphConvolution(int(3*nfeat), 1)
        self.dropout = dropout

    def forward(self, z , n, adj):

        out = self.fc(z)
        out = torch.tanh(self.bnfc(out))
        out = out.view(-1, 1, self.nver, int(4*self.nfeat))
        out = torch.cat((out,n),3)
        out = torch.tanh(self.bn1(self.gc1(out, adj)))
        out = F.dropout(out, self.dropout, training=self.training)
        out = torch.tanh(self.gc2(out, adj))

        return out

class GAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, l1, n1, adj1, n2, adj2):
        z = self.encoder(l1, n1, adj1)
        # if adj2 is not None:
        #     adj2 = adj2.permute(0, 1, 3, 2)
        recons = self.decoder(z, n2, adj2)
        return recons


class Discriminator(nn.Module):
    def __init__(self, nfeat, nver, dropout):
        super(Discriminator, self).__init__()
        self.nver = nver
        self.nfeat = nfeat
        self.gc1 = GraphConvolution(nfeat, 2*nfeat)
        self.gc2 = GraphConvolution(2*nfeat, 3*nfeat)
        self.bn1 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(3*nver*nfeat, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.dropout = dropout

    def forward(self,x1,adj1):
        out1 = torch.tanh(self.bn1(self.gc1(x1, adj1)))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = torch.tanh(self.gc2(out1, adj1))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = out1.view(-1, int(3*self.nver*self.nfeat))
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out1 = torch.sigmoid(out1)
        return out1


class LSDiscriminator(nn.Module):
    def __init__(self, nfeat, nver, dropout):
        super(LSDiscriminator, self).__init__()
        self.nver = nver
        self.nfeat = nfeat
        self.gc1 = GraphConvolution(nfeat, 2*nfeat)
        self.bn1 = nn.BatchNorm2d(1)
        self.gc2 = GraphConvolution(2*nfeat, 3*nfeat)
        self.bn2 = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(3*nver*nfeat, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1)
        self.dropout = dropout

    def forward(self,x1,adj1):
        out1 = torch.tanh(self.bn1(self.gc1(x1, adj1)))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = torch.tanh(self.bn2(self.gc2(out1, adj1)))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = out1.view(-1, int(3*self.nver*self.nfeat))
        out1 = self.bn3(self.fc1(out1))
        out1 = self.fc2(out1)
        return out1

class ZDiscriminator(nn.Module):
    def __init__(self, z,dropout):
        super(ZDiscriminator, self).__init__()

        self.z = z
        self.fc1 = nn.Linear(self.z, self.z)
        self.bn1 = nn.BatchNorm1d(self.z)
        self.fc2 = nn.Linear(self.z, self.z)
        self.bn2 = nn.BatchNorm1d(self.z)
        self.fc3 = nn.Linear(self.z, self.z)
        self.bn3 = nn.BatchNorm1d(self.z)
        self.fc4 = nn.Linear(self.z, 1)
        self.dropout = dropout

    def forward(self,x1):
        x1 = x1.view(-1, self.z)
        out1 = torch.tanh(self.bn1(self.fc1(x1)))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = torch.tanh(self.bn2(self.fc2(out1)))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = torch.tanh(self.bn3(self.fc3(out1)))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = self.fc4(out1)
        out1 = torch.sigmoid(out1)
        return out1        
