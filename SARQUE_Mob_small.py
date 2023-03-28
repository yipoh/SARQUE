from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.data
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
import torchvision.models as models
from prefetch_generator import BackgroundGenerator
from torch.nn.parameter import Parameter
import scipy.sparse as sp


class Brightness_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(Brightness_Model, self).__init__()

        self.drop_prob = (1 - keep_probability)
        self.fc1 = nn.Linear(inputsize, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1(out_a)
        out_a = self.relu1(out_a)
        out_a = self.drop1(out_a)
        out_a = self.fc2(out_a)
        # out_a = self.bn2_2(out_a)
        out_a = self.relu2(out_a)
        out_a = self.drop2(out_a)
        out_a = self.fc3(out_a)

        return out_a, f_out_a


class Colorfulness_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(Colorfulness_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1 = nn.Linear(inputsize, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1(out_a)
        out_a = self.relu1(out_a)
        out_a = self.drop1(out_a)
        out_a = self.fc2(out_a)
        out_a = self.relu2(out_a)
        out_a = self.drop2(out_a)
        out_a = self.fc3(out_a)

        return out_a, f_out_a


class Contrast_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(Contrast_Model, self).__init__()

        self.drop_prob = (1 - keep_probability)
        self.fc1 = nn.Linear(inputsize, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1(out_a)
        out_a = self.relu1(out_a)
        out_a = self.drop1(out_a)
        out_a = self.fc2(out_a)
        # out_a = self.bn2_2(out_a)
        out_a = self.relu2(out_a)
        out_a = self.drop2(out_a)
        out_a = self.fc3(out_a)

        return out_a, f_out_a

class Sharpness_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(Sharpness_Model, self).__init__()
        self.drop_prob = (1 - keep_probability)
        self.fc1 = nn.Linear(inputsize, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)  # 第一层
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)  # 第二层
        self.fc3 = nn.Linear(64, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1(out_a)
        out_a = self.relu1(out_a)
        out_a = self.drop1(out_a)
        out_a = self.fc2(out_a)
        # out_a = self.bn2_2(out_a)
        out_a = self.relu2(out_a)
        out_a = self.drop2(out_a)
        out_a = self.fc3(out_a)

        return out_a, f_out_a


class Noisiness_Model(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(Noisiness_Model, self).__init__()

        self.drop_prob = (1 - keep_probability)

        self.fc1 = nn.Linear(inputsize, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)  # 第一层
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)  # 第二层
        self.fc3 = nn.Linear(64, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1(out_a)
        out_a = self.relu1(out_a)
        out_a = self.drop1(out_a)
        out_a = self.fc2(out_a)
        # out_a = self.bn2_2(out_a)
        out_a = self.relu2(out_a)
        out_a = self.drop2(out_a)
        out_a = self.fc3(out_a)

        return out_a, f_out_a


#######################   GCN

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum==0]=0.0000001
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.00000001
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*2)

        self.dropout = dropout
        self.fc1 = nn.Linear(nhid *2* 6, 1)


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = x.view(x.shape[0],1,-1)
        #print(x.ndim)
        if x.ndim==2:
            x = x.view(1,-1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = x.reshape((x.shape[0], -1))
        return x


class AesModel1(nn.Module):
    def __init__(self, keep_probability, inputsize):

        super(AesModel1, self).__init__()
        self.drop_prob = 0.5
        self.fc1_1 = nn.Linear(inputsize, 256)
        self.bn1_1 = nn.BatchNorm1d(256)
        self.relu1_1 = nn.PReLU()
        self.drop1_1 = nn.Dropout(self.drop_prob)
        self.fc2_1 = nn.Linear(256, 64)
        self.relu2_1 = nn.PReLU()
        self.drop2_1 = nn.Dropout(p=self.drop_prob)
        self.fc3_1 = nn.Linear(64, 9)
        self.soft = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """

        out_a = self.fc1_1(x)
        f_out_a = out_a.clone()
        out_a = self.bn1_1(out_a)
        out_a = self.relu1_1(out_a)
        out_a = self.drop1_1(out_a)
        out_a = self.fc2_1(out_a)
        out_a = self.relu2_1(out_a)
        out_a = self.drop2_1(out_a)
        out_a = self.fc3_1(out_a)
        out_a = self.soft(out_a)
        return out_a, f_out_a


class convNetSce(nn.Module):
    #constructor
    def __init__(self,resnet,aesnet1):
        super(convNetSce, self).__init__()
        #defining layers in convnet
        self.resnet=resnet
        self.AesNet1=aesnet1



    def forward(self, x):
        x=self.resnet(x)
        out_a, f_out_a = self.AesNet1(x)

        return out_a, f_out_a


############### convNet

class convNet(nn.Module):
    #constructor
    def __init__(self, resnet, Brightness_net, Colorfulness_net, Contrast_net, Sharpness_net, Noisiness_net):
        super(convNet, self).__init__()
        self.resnet=resnet
        self.Brightness_Net=Brightness_net
        self.Colorfulness_Net=Colorfulness_net
        self.Contrast_Net=Contrast_net
        self.Sharpness_Net=Sharpness_net
        self.Noisiness_Net=Noisiness_net



    def forward(self, x_img):
        f1=self.resnet(x_img)
        Brightness_x, Brightness_f = self.Brightness_Net(f1)
        Colorfulness_x, Colorfulness_f = self.Colorfulness_Net(f1)
        Contrast_x, Contrast_f = self.Contrast_Net(f1)
        Sharpness_x, Sharpness_f = self.Sharpness_Net(f1)
        Noisiness_x, Noisiness_f = self.Noisiness_Net(f1)
        return Brightness_f, Colorfulness_f, Contrast_f, Sharpness_f, Noisiness_f, Brightness_x, Colorfulness_x, Contrast_x, Sharpness_x, Noisiness_x

############### convNet2

class convNet2(nn.Module):

    def __init__(self, attr_net, scene_net, gcn_net):
        super(convNet2, self).__init__()
        #defining layers in convnet
        self.AttrNet=attr_net
        self.ScenceNet = scene_net
        self.GCNNet = gcn_net



    def forward(self, x_img):

        x0, Scence_f=self.ScenceNet(x_img)
        Brightness_f, Colorfulness_f, Contrast_f, Sharpness_f, Noisiness_f,Brightness_x, Colorfulness_x, Contrast_x, Sharpness_x, Noisiness_x = self.AttrNet(x_img)

        temp = torch.zeros(Scence_f.shape[0], 6, Scence_f.shape[1])
        for num in range(Scence_f.shape[0]):
            temp[num : :] = torch.stack((Scence_f[num,:], Sharpness_f[num,:], Brightness_f[num,:], Colorfulness_f[num,:], Contrast_f[num,:], Noisiness_f[num,:]), 0)
        edges_unordered = np.genfromtxt("cora.cites", dtype=np.int32)  # 读入边的信息
        adj = np.zeros((6, 6))
        for [q, p] in edges_unordered:
            adj[q - 1, p - 1] = 1
        adj = torch.from_numpy(adj)
        adj = normalize(adj)
        adj = torch.from_numpy(adj)
        adj = adj.clone().float()
        adj = adj.to(device)
        temp=temp.to(device)
        out_a= self.GCNNet(temp, adj)
        return out_a, Brightness_x, Colorfulness_x, Contrast_x, Sharpness_x, Noisiness_x


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)


    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]), self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]



def mytest(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            torch.random.manual_seed(len(test_loader) * epoch + batch_idx)
            rd_ps = torch.randint(20, (3,))
            data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
            if rd_ps[1] < 10:
                data = torch.flip(data, dims=[3])
            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            Quality, Brightness, Colorfulness, Contrast, Sharpness, Noisiness = model(data)
            Quality= Quality[:, 0].detach().cpu().numpy()
            Brightness=Brightness[:, 0].detach().cpu().numpy()
            Colorfulness=Colorfulness[:, 0].detach().cpu().numpy()
            Contrast=Contrast[:, 0].detach().cpu().numpy()
            Sharpness=Sharpness[:, 0].detach().cpu().numpy()
            Noisiness=Noisiness[:, 0].detach().cpu().numpy()
            print()
            print('#: ',batch_idx)
            print('Quality: ', Quality[0])
            print('Brightness: ', Brightness[0])
            print('Colorfulness: ', Colorfulness[0])
            print('Contrast: ', Contrast[0])
            print('Sharpness: ', Sharpness[0])
            print('Noisiness: ', Noisiness[0])



def main():

    batch_size = 1
    num_workers_test = 0

    all_data = np.load('SPAQ_10images.npz')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(-1, 1)

    del all_data
    test_dataset = Mydataset(X, Y)
    test_loader = DataLoaderX(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_test,
                              pin_memory=True)

    ###### Mob-Attr
    model_ft = models.mobilenet_v3_small(pretrained=True)
    num_ftrs = 1000
    Brightness_net = Brightness_Model(0.5, num_ftrs)
    Colorfulness_net = Colorfulness_Model(0.5, num_ftrs)
    Contrast_net = Contrast_Model(0.5, num_ftrs)
    Sharpness_net = Sharpness_Model(0.5, num_ftrs)
    Noisiness_net = Noisiness_Model(0.5, num_ftrs)
    Attrmodel = convNet(resnet=model_ft, Brightness_net=Brightness_net, Colorfulness_net=Colorfulness_net,
                        Contrast_net=Contrast_net, Sharpness_net=Sharpness_net, Noisiness_net=Noisiness_net)
    Attrmodel = Attrmodel.to(device)


    ###### Mob-Sce
    pretrained_dict = models.mobilenet_v3_small(pretrained=True)
    num_ftrs = 1000
    net1 = AesModel1(0.5, num_ftrs)
    modelSce = convNetSce(resnet=pretrained_dict, aesnet1=net1)
    modelSce = modelSce.to(device)

    #####################  GCN
    model_GCN = GCN(nfeat=256, nhid=1024, dropout=0.5).to(device)

################### Fusion_convNet2
    model_all = convNet2(attr_net=Attrmodel, scene_net=modelSce, gcn_net=model_GCN)
    model_all = model_all.to(device)
    pretrained_dict = torch.load("Mob_SPAQ_small.pt")
    torch.cuda.empty_cache()
    model_all = model_all.to(device)
    model_dict = model_all.state_dict()  # 读出搭建的网络的参数，以便后边更新之后初始化
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_all.load_state_dict(model_dict)



    mytest(model_all, test_loader, 0, device, 0)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")
    main()

