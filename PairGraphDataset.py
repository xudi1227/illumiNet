'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-09-01 19:55:37
@LastEditTime: 2019-09-02 20:43:38
@LastEditors: Please set LastEditors
'''
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import load_color_data,load_adj_data,load_normal_data, get_pixel_mask,load_light_normal_adj,load_color_normal_adj,load_nonself_adj
import os
import re
import PIL


class GraphDataset(Dataset):
    def __init__(self, path_dir,num_pose):

        self.filename = self.read_file(path_dir)
        self.path_dir = path_dir
        self.len = len(self.filename)
        self.num_pose = num_pose

    def __getitem__(self, i):

        index1 = int(i % self.len)

        fname1 = self.filename[index1]
        # find another sample in a different pose under same illumination 
        fname2 = re.sub(r'pos[0-9]+',"pos"+str(np.random.randint(self.num_pose)),fname1)
        fpath1 = os.path.join(self.path_dir, fname1)
        fpath2 = os.path.join(self.path_dir, fname2)

        light1,normal1,adj1 = load_light_normal_adj(fpath1, normalization=True)
        light2,normal2,adj2 = load_light_normal_adj(fpath2, normalization=True)

        light1 = transforms.ToTensor()(light1)
        normal1 = transforms.ToTensor()(normal1)
        adj1 = transforms.ToTensor()(adj1)
        light2 = transforms.ToTensor()(light2)
        normal2 = transforms.ToTensor()(normal2)
        adj2 = transforms.ToTensor()(adj2)

        #print(fname)

        return light1,normal1,adj1,light2,normal2,adj2,fname1,fname2

    def __len__(self):

        data_len = len(self.filename)

        return data_len

    def read_file(self,path):

        filename = []
        if os.path.isdir(path):
            filelist = os.listdir(path)
            for f in filelist:
                filename.append(f)
        #print(len(filename))
        return filename

class LightColorDataset(Dataset):
    def __init__(self,path_A_dir,path_B_dir,name_A,name_B,name_attri_A,name_attri_B):

        self.filenameA = self.read_file(path_A_dir)
        self.filenameB = self.read_file(path_B_dir)
        self.path_A_dir = path_A_dir
        self.path_B_dir = path_B_dir
        self.name_object_A = name_A
        self.name_object_B = name_B
        self.name_attri_A = name_attri_A
        self.name_attri_B = name_attri_B

        if(len(self.filenameA) ==len(self.filenameB)):
            self.len = len(self.filenameA)
        else:
            print("The number of datasetA and datasetB are inconsistent!!!")

    def __getitem__(self, i):
        
        if self.len is None:
            print("The number of datasetA and datasetB are inconsistent!!!")
            return "error"

        index = int(i % self.len)

        fname_A = self.filenameA[index]
        fname_B = fname_A.replace(self.name_object_A,self.name_object_B)
        fname_B = fname_B.replace(self.name_attri_A,self.name_attri_B)
        fpath_A = os.path.join(self.path_A_dir, fname_A)
        fpath_B = os.path.join(self.path_B_dir, fname_B)
        
        light1,normal1,adj1 = load_light_normal_adj(fpath_A, normalization=True)
        if(self.name_object_A == self.name_object_B):
            color = load_color_data(fpath_B, normalization=True)
            nonselfadj = load_nonself_adj(fpath_B)

            light1 = transforms.ToTensor()(light11)
            normal1 = transforms.ToTensor()(normal1)
            adj1 = transforms.ToTensor()(adj1)
            nonselfadj = transforms.ToTensor()(nonselfadj)

            return light1, normal1, adj1, color, fname_A, fname_B,nonselfadj
        else:
            color,normal2,adj2 = load_color_normal_adj(fpath_B, normalization=True)
            nonselfadj = load_nonself_adj(fpath_B)

            light1 = transforms.ToTensor()(light11)
            normal1 = transforms.ToTensor()(normal1)
            adj1 = transforms.ToTensor()(adj1)
            color = transforms.ToTensor()(color)
            normal2 = transforms.ToTensor()(normal2)
            adj2 = transforms.ToTensor()(adj1)
            nonselfadj = transforms.ToTensor()(nonselfadj)

            return light1, normal1, adj1, color, normal2, adj2, fname_A, fname_B,nonselfadj
            

        

    def __len__(self):
        if(len(self.filenameA) ==len(self.filenameB)):
            data_len = len(self.filenameA)
        else:
            print("The number of datasetA and datasetB are inconsistent!!!")
        
        return data_len

    def read_file(self,path):

        filename = []
        if os.path.isdir(path):
            filelist = os.listdir(path)
            for f in filelist:
                filename.append(f)
        #print(len(filename))
        return filename    


class PairColorDataset(Dataset):
    def __init__(self,path_A_dir,path_B_dir):

        self.filenameA = self.read_file(path_A_dir)
        self.filenameB = self.read_file(path_B_dir)
        self.path_A_dir = path_A_dir
        self.path_B_dir = path_B_dir
        
        if(len(self.filenameA) <=len(self.filenameB)):
            self.len = len(self.filenameA)
        else:
            print("The number of datasetA and datasetB are inconsistent!!!")

    def __getitem__(self, i):
        
        if self.len is None:
            print("The number of datasetA and datasetB are inconsistent!!!")
            return "error"

        index = int(i % self.len)

        fname_A = self.filenameA[index]
        fpath_A = os.path.join(self.path_A_dir, fname_A)
        fpath_B = os.path.join(self.path_B_dir, fname_A)

        data_A = load_color_data(fpath_A, normalization=True)
        data_B = load_color_data(fpath_B, normalization=True)
        

        features_A = transforms.ToTensor()(data_A)
        features_B = transforms.ToTensor()(data_B)
    
        return features_A,features_B,fname_A

    def __len__(self):
        if(len(self.filenameA) <=len(self.filenameB)):
            data_len = len(self.filenameA)
        else:
            print("The number of datasetA and datasetB are inconsistent!!!")
        
        return data_len

    def read_file(self,path):

        filename = []
        if os.path.isdir(path):
            filelist = os.listdir(path)
            for f in filelist:
                filename.append(f)
        #print(len(filename))
        return filename


class PairGLightColorDataset(Dataset):
    def __init__(self, path_A_dir,path_B_dir,name_A,name_B,path_B_color,name_attri_A,name_attri_B,num_pose):

        self.filenameA = self.read_file(path_A_dir)
        self.filenameB = self.read_file(path_B_dir)
        self.filenameB_color = self.read_file(path_B_color)
        self.path_A_dir = path_A_dir
        self.path_B_dir = path_B_dir
        self.path_B_color = path_B_color
        self.name_A = name_A
        self.name_B = name_B
        self.attri_A = name_attri_A
        self.attri_B = name_attri_B
        self.num_pose = num_pose

        if(len(self.filenameA) ==len(self.filenameB)):
            self.len = len(self.filenameA)
        else:
            print("The number of datasetA and datasetB are inconsistent!!!")

    def __getitem__(self, i):
        
        if self.len is None:
            print("The number of datasetA and datasetB are inconsistent!!!")
            return "error"

        index = int(i % self.len)

        fname_A = self.filenameA[index]
        fname_B = fname_A.replace(self.name_A,self.name_B)
        fname_B = re.sub(r'pos[0-9]+',"pos"+str(np.random.randint(self.num_pose)),fname_B)
        fname_B_color = fname_B.replace(self.attri_A,self.attri_B)

        fpath_A = os.path.join(self.path_A_dir, fname_A)
        fpath_B = os.path.join(self.path_B_dir, fname_B)
        fpath_B_color = os.path.join(self.path_B_color, fname_B_color)
        # print(fpath_B)
        light1,normal1,adj1 = load_light_normal_adj(fpath_A, normalization=True)
        light2,normal2,adj2 = load_light_normal_adj(fpath_B, normalization=True)

        features_B_color = load_color_data(fpath_B_color, normalization=True)
        nonselfadj = load_nonself_adj(fpath_B_color)

        light1 = transforms.ToTensor()(light1)
        normal1 = transforms.ToTensor()(normal1)
        adj1 = transforms.ToTensor()(adj1)
        light2 = transforms.ToTensor()(light2)
        normal2 = transforms.ToTensor()(normal2)
        adj2 = transforms.ToTensor()(adj2)

        features_B_color = transforms.ToTensor()(features_B_color)
        nonselfadj = transforms.ToTensor()(nonselfadj)

        return light1, normal1, adj1, light2, normal2, adj2, features_B_color, nonselfadj, fname_A, fname_B

    def __len__(self):
        if(len(self.filenameA) ==len(self.filenameB)):
            data_len = len(self.filenameA)
        else:
            print("The number of datasetA and datasetB are inconsistent!!!")
        
        return data_len

    def read_file(self,path):

        filename = []
        if os.path.isdir(path):
            filelist = os.listdir(path)
            for f in filelist:
                filename.append(f)
        #print(len(filename))
        return filename        




class CustomDataset(Dataset):
    def __init__(self, path_dir,path_target_mesh):

        self.filename = self.read_file(path_dir)
        self.path_dir = path_dir
        self.len = len(self.filename)
        self.target_normal = transforms.ToTensor()(load_normal_data(path_target_mesh))
        self.target_adj = transforms.ToTensor()(load_adj_data(path_target_mesh))


    def __getitem__(self, i):

        index = int(i % self.len)

        fname = self.filename[index]
        fpath = os.path.join(self.path_dir, fname)

        light,normal,adj = load_light_normal_adj(fpath, normalization=True)

        light = transforms.ToTensor()(light)
        normal = transforms.ToTensor()(normal)
        adj = transforms.ToTensor()(adj)

        #print(fname)

        return light, normal, adj, self.target_normal, self.target_adj, fname

    def __len__(self):

        data_len = len(self.filename)

        return data_len

    def read_file(self,path):

        filename = []
        if os.path.isdir(path):
            filelist = os.listdir(path)
            for f in filelist:
                filename.append(f)
        #print(len(filename))
        return filename
