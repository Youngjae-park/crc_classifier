#-*- coding: utf-8 -*-
"""
Created on 2020. 12. 03. (Thur) 12:54:43 KST

@author: youngjae-park
"""

import numpy as np
import os
import time
import queue
import utils

np.random.seed(2)


class IP:
    def __init__(self, to_use='Train_3cls', validation=False):
        self.to_use = to_use

        if to_use == 'Train_3cls':
            path = '../dataset/train_3cls'
            data_list = [os.path.join(path, 'whole-{}.hdf5'.format(i)) for i in range(tot_num)]
        elif to_use == 'Train_2cls':
            path = '../dataset/train_2cls'
            data_list = []
            for x in os.listdir(path):
                data_list.append(os.path.join(path, x))
            #data_list = os.listdir(path)
        elif to_use == 'Test':
            path = '../dataset/test'
            data_list = None # Change it later
        
        self.data_list = np.asarray(data_list)
        #print(type(self.data_list))
        #print('DATA_LIST: ', self.data_list) 
        if validation:
            one_idx = np.asarray([0,4,5,11,19,23,29,31,33,41,43,46])
            np.random.shuffle(one_idx)
            zero_idx = np.arange(47)
            zero_idx = np.setdiff1d(zero_idx, one_idx)
            np.random.shuffle(zero_idx)
            #print('one_idx {}'.format(one_idx))
            #print('zero_idx {}'.format(zero_idx))
            validation_idx = np.union1d(one_idx[:3], zero_idx[:9])
            np.random.shuffle(validation_idx)
            self.valid_list = self.data_list[validation_idx].tolist()
            all_idx = np.arange(47)
            np.random.shuffle(all_idx)
            tr_idx = np.setdiff1d(all_idx, validation_idx)
            np.random.shuffle(tr_idx)
            self.train_list = self.data_list[tr_idx].tolist()
            #print('self.train_list', self.train_list)
            #print('self.valid_list', self.valid_list)
        else:
            all_idx = np.arange(47)
            np.random.shuffle(all_idx)
            #print('order_idx {}'.format(all_idx))
            self.train_list = self.data_list[all_idx].tolist()
            #print(self.train_list)
        
        #self.train_queue = queue.Queue()
        #self.validation_queue = queue.Queue()
    
    @staticmethod
    def _normalize(arr):
        return arr/255.
        #return (arr - arr.min(axis=3, keepdims=True)) / (arr.max(axis=3, keepdims=True) - arr.min(axis=3, keepdims=True))

    def load_train(self, shuffle=True, batch_size=32):
        if self.train_list:
            tr_name = self.train_list.pop()
            coord, matx, laby = utils.load_hdf5(tr_name, using='Train')
            if shuffle:
                idx = np.arange(coord.shape[0])
                np.random.shuffle(idx)
            matx = self._normalize(matx)
            #print(coord)
            #print(matx)
            #print(laby)
            return coord[idx], matx[idx], laby[idx]
        else:
            print('train_list is empty')
            return None

    def load_validation(self, shuffle=True):
        if self.valid_list:
            val_name = self.valid_list.pop()
            coord, matx, laby = utils.load_hdf5(val_name, using='Validation')
            if shuffle:
                idx = np.arange(coord.shape[0])
                np.random.shuffle(idx)
            matx = self._normalize(matx)
            return coord[idx], matx[idx], laby[idx]
        else:
            print('valid_list is empty')
            return None



    # def init_producer(self, shuffle=True, batch_size=32):


if __name__ == '__main__':
    #ip = IP(to_use='Train_3cls', validation=True)
    #ip = IP(to_use='Train_2cls', validation=True)
    ip = IP(to_use='Train_2cls', validation=True) #False)
    ip.load_train()

