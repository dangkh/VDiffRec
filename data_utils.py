import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item

def subdata_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0

    train_dict = {}
    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1

    valid_dict = {}
    for uid, iid in valid_list:
        if uid not in valid_dict:
            valid_dict[uid] = []
        valid_dict[uid].append(iid)
    
    test_dict = {}
    for uid, iid in test_list:
        if uid not in test_dict:
            test_dict[uid] = []
        test_dict[uid].append(iid)
    
    return train_dict, valid_dict, test_dict, n_user, n_item

class SubData(Dataset):
    def __init__(self, train_path, valid_path, test_path, num_sub=1000):
        super(SubData, self).__init__()
        self.train_dict, self.valid_dict, self.test_dict, \
            self.num_user, self.num_item = subdata_load(train_path, valid_path, test_path)
        self.num_sub = num_sub
        
        self.item_set = set(range(0, self.num_item))

        self.all_user = [i for i in range(self.num_user)]

        self.val_list, self.val_gt = self.get_val(self.valid_dict)
        self.test_list, self.test_gt = self.get_test(self.test_dict)

    def get_val(self, data):
        # data: ground truth
        val_list = [[] for _ in range(self.num_user)]
        gt_list = [[] for _ in range(self.num_user)]

        for uid in data:
            val_list[uid].extend(data[uid])
            gt_list[uid].extend([i for i in range(len(data[uid]))])
            try:
                a = self.item_set - set(self.train_dict[uid])  # mask train set
            except:
                if uid not in self.train_dict:
                    print("User not found.")
                print("Error!")
            m = np.random.choice(np.array(list(a)), self.num_sub-len(val_list[uid]), replace=False)
            val_list[uid].extend(m)
        
        for i in range(len(val_list)):
            if len(val_list[i]) == 0:
                val_list[i] = [0] * self.num_sub
        
        val_list = torch.LongTensor(val_list)
        return val_list, gt_list
    
    def get_test(self, data):
        test_list = [[] for _ in range(self.num_user)]
        gt_list = [[] for _ in range(self.num_user)]

        for uid in data:
            test_list[uid].extend(data[uid])
            gt_list[uid].extend([i for i in range(len(data[uid]))])
            try:
                m = np.random.choice(np.array(list(self.item_set-set(self.train_dict[uid])-set(self.valid_dict[uid]))), self.num_sub-len(test_list[uid]), replace=False)
            except:
                m = np.random.choice(np.array(list(self.item_set-set(self.train_dict[uid]))), self.num_sub-len(test_list[uid]), replace=False)
            test_list[uid].extend[m]

        for i in range(len(test_list)):
            if len(test_list[i]) == 0:
                test_list[i] = [0] * self.num_sub
        
        test_list = torch.LongTensor(test_list)
        return test_list, gt_list
            

class DataDiffusion(Dataset):
    def __init__(self, data, embed, maxItem):
        self.data = data
        self.embed = embed
        self.userEmb = []
        self.userEmbMean = []
        self.user_numInteract = []
        self.userEmbSum = []
        self.maxItem = maxItem
        sfm = torch.nn.Softmax(dim = 1)
        self.label = sfm(data)
        for ii in range(len(self.data)):
            fulEm, emb1, numI, sumEmb = self.index2itemEm(data[ii])
            self.user_numInteract.append(numI)
            self.userEmb.append(fulEm)
            self.userEmbMean.append(emb1)
            self.userEmbSum.append(sumEmb)
        self.cumNumInteract = [0]
        for ii in range(len(self.user_numInteract)):
            self.cumNumInteract.append(self.cumNumInteract[-1] + self.user_numInteract[ii])
        self.userEmb = torch.vstack(self.userEmb)

    def index2itemEm(self, itemIndx):
        output = []
        clickedItem = torch.where(itemIndx == 1)[0]
        index = torch.randperm(len(clickedItem))
        clickedItem = clickedItem[index]
        counter = 0
        for ii in clickedItem:
            output.append(torch.reshape(self.embed[ii.item()], (1,-1)))
            counter += 1
            if counter > (self.maxItem-1):
                break
        emb_wo_comp = torch.vstack(output).mean(0)
        sumEmb = torch.vstack(output).sum(0) 
        # compensationNum = self.maxItem -counter
        # compensationFeat = torch.zeros((compensationNum,64))
        # output.append(compensationFeat)
        return torch.vstack(output), emb_wo_comp, len(clickedItem), sumEmb

    def __getitem__(self, index):
        # l, r = self.cumNumInteract[index], self.cumNumInteract[index+1]
        numI = self.user_numInteract[index]
        maskIdx = np.random.randint(0, numI)
        maskEmb = self.userEmb[maskIdx]
        embed = (self.userEmbSum[index] - maskEmb) / (numI-1)
        item = self.data[index]
        label = self.label[index]
        return item, [embed, maskEmb], label

    def __len__(self):
        return len(self.data)
