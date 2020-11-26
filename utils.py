'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-08-23 21:54:44
@LastEditTime: 2019-09-02 20:53:16
@LastEditors: Please set LastEditors
'''

import numpy as np
import scipy.sparse as sp
import torch
from numpy import linalg as la
import time
import logging
from PIL import Image
import numpy as np


'''
@description: 读取off文件，跳过前三行，然后前面是n×6的矩阵 n是点个数（x y z ol1 ol2 ol3），后面是m×4的矩阵 m是面个数（3 index1 index2 index3）
@param {type} 
@return: A : n*3的矩阵，每个点3维特征；
'''


#load all data from xx.ply
def load_light_normal_adj(path,normalization):
    with open(path, 'r') as f:
        lines = f.readlines()
    nver = int(lines[3].split(' ')[2])
    del lines[0:17]

    normal = np.zeros((nver, 3), dtype=float)
    light = np.zeros((nver, 3), dtype=float)
    adj = np.zeros((nver, nver), dtype=float)

    i = 0

    for line in lines:
        if i < nver:
            value = line.strip('\n').split(' ')
            #print(value[0])
            #print(value)
            normal[i][0] = value[3]  # value[3:6]
            normal[i][1] = value[4]
            normal[i][2] = value[5]

            light[i][0] = value[6]  # value[6:9]
            light[i][1] = value[7]
            light[i][2] = value[8]
            i += 1
        else:
            value = line.strip('\n').split(' ')   # 带自连接的邻接矩阵
            index1=int(value[1])
            index2=int(value[2])
            index3=int(value[3])
            adj[index1][index1] = 1
            adj[index1][index2] = 1
            adj[index1][index3] = 1
            adj[index2][index1] = 1
            adj[index2][index2] = 1
            adj[index2][index3] = 1
            adj[index3][index1] = 1
            adj[index3][index2] = 1
            adj[index3][index3] = 1
    
    for i in range(0,nver):
        if adj[i][i] ==0:
            adj[i][i] =1

    Dsum=np.sum(adj,axis=1)
    V = np.diag(Dsum**(-0.5))
    adj = V*adj*V

    if normalization == True:
        light = light/255
    
    return light,normal,adj



#load all data from  xx.ply
def load_color_normal_adj(path,normalization):
    with open(path, 'r') as f:
        lines = f.readlines()
    nver = int(lines[3].split(' ')[2])
    del lines[0:17]

    normal = np.zeros((nver, 3), dtype=float)
    color = np.zeros((nver, 1), dtype=float)
    adj = np.zeros((nver, nver), dtype=float)

    i = 0

    for line in lines:
        if i < nver:
            value = line.strip('\n').split(' ')
            #print(value[0])
            #print(value)
            normal[i][0] = value[3]  # value[3:6]
            normal[i][1] = value[4]
            normal[i][2] = value[5]

            color[i][0] = value[6]  # value[6:9]
            i += 1
        else:
            value = line.strip('\n').split(' ')   # 带自连接的邻接矩阵
            index1=int(value[1])
            index2=int(value[2])
            index3=int(value[3])
            adj[index1][index1] = 1
            adj[index1][index2] = 1
            adj[index1][index3] = 1
            adj[index2][index1] = 1
            adj[index2][index2] = 1
            adj[index2][index3] = 1
            adj[index3][index1] = 1
            adj[index3][index2] = 1
            adj[index3][index3] = 1
    
    for i in range(0,nver):
        if adj[i][i] ==0:
            adj[i][i] =1

    Dsum=np.sum(adj,axis=1)
    V = np.diag(Dsum**(-0.5))
    adj = V*adj*V

    if normalization == True:
        color = color/255
    
    return color,normal,adj


def load_color_data(filepath, normalization=True):
    with open(path, 'r') as f:
        lines = f.readlines()
    nver = int(lines[3].split(' ')[2])
    del lines[0:17]

    color = np.zeros((nver, 1), dtype=float)

    i = 0

    for line in lines:
        if i < nver:
            value = line.strip('\n').split(' ')
            color[i][0] = value[6]  # value[6:9]
            i += 1
        else:
            break
    if normalization == True:
        color = color/255
    
    return color

def load_adj_data(filepath):

    with open(path, 'r') as f:
        lines = f.readlines()
    nver = int(lines[3].split(' ')[2])
    del lines[0:17]
    adj = np.zeros((nver, nver), dtype=float)

    i = 0

    for line in lines:
        if i >= nver:
            value = line.strip('\n').split(' ')   # 带自连接的邻接矩阵
            index1=int(value[1])
            index2=int(value[2])
            index3=int(value[3])
            adj[index1][index1] = 1
            adj[index1][index2] = 1
            adj[index1][index3] = 1
            adj[index2][index1] = 1
            adj[index2][index2] = 1
            adj[index2][index3] = 1
            adj[index3][index1] = 1
            adj[index3][index2] = 1
            adj[index3][index3] = 1
    
    for i in range(0,nver):
        if adj[i][i] ==0:
            adj[i][i] =1

    Dsum=np.sum(adj,axis=1)
    V = np.diag(Dsum**(-0.5))
    adj = V*adj*V
    return torch.Tensor(adj)


def load_nonself_adj(filepath):

    with open(path, 'r') as f:
        lines = f.readlines()
    nver = int(lines[3].split(' ')[2])
    del lines[0:17]
    adj = np.zeros((nver, nver), dtype=float)

    i = 0

    for line in lines:
        if i >= nver:
            value = line.strip('\n').split(' ')
            index1=int(value[1])
            index2=int(value[2])
            index3=int(value[3])
            adj[index1][index1] = 1
            adj[index1][index2] = 1
            adj[index1][index3] = 1
            adj[index2][index1] = 1
            adj[index2][index2] = 1
            adj[index2][index3] = 1
            adj[index3][index1] = 1
            adj[index3][index2] = 1
            adj[index3][index3] = 1

    Dsum=np.sum(adj,axis=1)
    V = np.diag(Dsum**(-0.5))
    adj = V*adj*V
    return torch.Tensor(adj)


def writeOFFfile(reconst_filename,pointlist,source_filename):

    with open(source_filename, 'r') as f:
        lines = f.readlines()

    nver = int(lines[3].split(' ')[2])
    nface = int(lines[14].split(' ')[2])
    del lines[0:17]

    face = lines[len(pointlist):]
    A = np.zeros((nver, 3), dtype=float)
    for i in range(0,nver):
        value = lines[i].strip('\n').split(' ')
        #print(value[0])
        A[i][0] = value[0]  # value[3:6]
        A[i][1] = value[1]
        A[i][2] = value[2]
    
    with open(reconst_filename,'w') as f:
        f.write("COFF\n")
        f.writelines(str(len(pointlist))+' '+str(nface)+' '+'0\n')
        for i in range(0,len(pointlist)):
            f.writelines(str(A[i][0])+' '+str(A[i][1])+' '+str(A[i][2])+' '
            +str(int(pointlist[i][0]))+' '+str(int(pointlist[i][1]))+' '+str(int(pointlist[i][2]))+'\n')
        for line in face:
            f.writelines(line)

def writeColorOFFfile(reconst_filename,pointlist,source_filename):
    with open(source_filename, 'r') as f:
        lines = f.readlines()

    nver = int(lines[3].split(' ')[2])
    nface = int(lines[14].split(' ')[2])
    del lines[0:17]

    face = lines[len(pointlist):]
    A = np.zeros((nver, 3), dtype=float)
    for i in range(0,nver):
        value = lines[i].strip('\n').split(' ')
        #print(value[0])
        A[i][0] = value[0]  # value[3:6]
        A[i][1] = value[1]
        A[i][2] = value[2]
    
    with open(reconst_filename,'w') as f:
        f.write("COFF\n")
        f.writelines(str(len(pointlist))+' '+str(nface)+' '+'0\n')
        for i in range(0,len(pointlist)):
            f.writelines(str(A[i][0])+' '+str(A[i][1])+' '+str(A[i][2])+' '
            +str(int(pointlist[i][0]))+' '+str(int(pointlist[i][0]))+' '+str(int(pointlist[i][0]))+'\n')
        for line in face:
            f.writelines(line)

def get_log(file_name):

    logger = logging.getLogger('train')  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level
 
    fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    #logger.addHandler(ch)
    return logger

#load normal data from .ply
def load_normal_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    nver = int(lines[3].split(' ')[2])
    del lines[0:17]

    A = np.zeros((nver, 3), dtype=float)
    for i in range(0,nver):
        value = lines[i].strip('\n').split(' ')
        #print(value[0])
        #print(value)
        A[i][0] = value[3]  # value[3:6]
        A[i][1] = value[4]
        A[i][2] = value[5]
    #print(A)

    return torch.Tensor(A)


# replace negative value into zero
def reprocess(x):

    for i in (x<0).nonzero():
        x[i[0]][i[1]] = 0

    return x


def smootherror(predicted,nonselfadj):
    
    size = predicted.size()
    out = torch.matmul(nonselfadj,predicted)
    out = out / nonselfadj.sum(2).view(size[0],size[1],size[2],1)
    out = out - predicted

    return torch.sum(abs(out))/(size[0]*size[1]*size[2]*size[3])


def get_pixel_mask(image_path):

    image = Image.open(image_path)
    width,height = image.size
    array= np.array(image)
    pixel_list = []
    for row in range(height):
        for col in range(width):
            if array[row][col][0] != 0 or array[row][col][1] != 0 or array[row][col][2] != 0:
                pixel_list.append(array[row][col][0])

    return np.array(pixel_list)
    
    # return lenth


if __name__ == "__main__":

   
    # x = torch.randn(2,3)
    # print(x)
    # for i in (x<0).nonzero():
    #     x[i[0]][i[1]] = 0
    # print(x)

    path = '/media/lz/Backup Plus/SpatiallyVarying/GT/FinalLightFile/Color/piexl/sphere_scene00_probe03_Color00.png'
    get_pixel_number_mask(path)
    # color = load_color_data(path,True)
    # print(color)