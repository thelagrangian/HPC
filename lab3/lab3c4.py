from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import utils
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributed as dist

class lab1Dataset(Dataset):

    def __init__(self, csv_file, img_dir,transform=None):
        self.tag = pd.read_csv(csv_file)
        self.img_dir   = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.tag)


    def __getitem__(self,i):
        img_name = os.path.join(self.img_dir,self.tag.iloc[i,0]) + '.jpg'

        start0 = time.time()
        with open(img_name,'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        end0 = time.time()
        t0 = end0 - start0

        start1 = time.time()
        if self.transform:
            transformed = self.transform(image)
        end1 =time.time()
        t1 = end1 - start1

        return { 'image': transformed, 'tag': self.tag.iloc[i,1],'t0':t0, 't1': t1 }


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n_feature, n_hidden1)
        self.fc2 = torch.nn.Linear(n_hidden1,n_hidden2)
        self.fc3 = torch.nn.Linear(n_hidden2,n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=0)
        return x
 

def run(rank, wsize):

    minibatch = 100
    num_epoch = 5
    num_worker= wsize - 1


    trans = transforms.Compose([transforms.Resize(32),transforms.ToTensor()])

    io_time_spent = 0.0
    aug_time_spent= 0.0


    train_set = lab1Dataset(csv_file='/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train.csv',
                            img_dir ='/scratch/am9031/CSCI-GA.3033-023/lab3/kaggleamazon/train-jpg/',
                            transform=trans)


    totsize = len(train_set)
    beginidx = (rank-1)*int(totsize/num_worker)
    endidx   = rank*int(totsize/num_worker)
    if rank == wsize - 1:
        endidx = totsize
    

    if rank > 0:
        train_set = [train_set[i] for i in range(len(train_set)) if i >= beginidx and i < endidx]
    else:
        train_set = [train_set[i] for i in range(len(train_set)) if i >= 0        and i < minibatch ]
    # print("{}: {}".format(rank,len(train_set)))

    train_loader = DataLoader(train_set, batch_size=minibatch, num_workers=1)

    net = Net(n_feature=3072, n_hidden1=1024, n_hidden2=256, n_output=17)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


    wgrp = dist.new_group(range(wsize)[1:])
    src    = torch.Tensor(1)
    looplimit = num_worker * int(int((totsize/num_worker) + minibatch - 1)/minibatch)
    loopcount = 0

    if rank == 0:
        for i, data in enumerate(train_loader):
            x = Variable(data['image'].view(len(data['tag']), -1))
            z = Variable(data['tag'])
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, z)
            loss.backward()
        for param in net.parameters():
            param.grad.data.zero_()



    dist.barrier()
    train_5_epoch_start =time.time()

    if rank == 0:
        for i in range(wsize)[1:]:
            for param in net.parameters():
                dist.send(param.data, i)
    else:
        for param in net.parameters():
            dist.recv(param.data, 0)


    dist.barrier()
    if rank == 0:
        for epoch in range(num_epoch):
            # print("{}: {}".format(rank, epoch))
            loopcount = 0
            while loopcount < looplimit:
                # print("rank 0: {}/{}".format(loopcount, looplimit))
                loopcount += 1
                src[0] = -1
                dist.recv(src)
                for param in net.parameters():
                    param.grad.data.zero_()
                # net.zero_grad()
                for param in net.parameters():
                    dist.recv(param.grad.data, int(src[0]))
                optimizer.step()
                for param in net.parameters():
                    dist.send(param.data, int(src[0]))
            # print("0: reached barrier")
            dist.barrier()
            for i in range(wsize)[1:]:
                for param in net.parameters():
                    dist.send(param.data, i)
    else:
        avgloss = 0.0
        wbatch  = len(train_set)
        num_batch = int((wbatch + minibatch - 1)/minibatch)

        for epoch in range(num_epoch):
            # print("{}: {}".format(rank, epoch))
            for i, data in enumerate(train_loader):
                # print("{}: {}, {}".format(rank, i,len(data['tag'])))
                x = Variable(data['image'].view(len(data['tag']), -1))
                z = Variable(data['tag'])
                optimizer.zero_grad()
                outputs = net(x)
                loss = criterion(outputs, z)
                loss.backward()
                if epoch == num_epoch - 1:
                    avgloss += loss.data
                src[0] = rank
                dist.send(src, 0)
                for param in net.parameters():
                    dist.send(param.grad.data, 0)
                for param in net.parameters():
                    dist.recv(param.data, 0)
            # print("{}: reached barrier".format(rank))
            dist.barrier()
            for param in net.parameters():
                dist.recv(param.data, 0);

        avgloss /= (1.0*num_batch)
        avgloss *= (1.0*wbatch)
        tensorl  = torch.tensor([avgloss, 1.0*wbatch])
        dist.all_reduce(tensorl, op = dist.reduce_op.SUM, group=wgrp)        

    dist.barrier()
    train_5_epoch_end = time.time()
    train_5_epoch_spent = train_5_epoch_end - train_5_epoch_start
    train_5_epoch_avg   = train_5_epoch_spent/float(num_epoch)
    if rank == 1:
        print("{}, {}".format(tensorl[0]/tensorl[1], train_5_epoch_avg))


if __name__ == "__main__":

    dist.init_process_group(backend='mpi')

    rank = dist.get_rank()
    wsize = dist.get_world_size()

    run(rank, wsize)
