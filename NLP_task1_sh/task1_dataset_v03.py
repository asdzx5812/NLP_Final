import os
import time
import unicodedata
import random
import string
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from alphabert_utils import split2words
from transformers import ElectraTokenizer, ElectraForTokenClassification


class task1ds_electra_train(Dataset):
    def __init__(self,ds,tokenizer,train=True,data_train_pos=None,data_train_neg=None,augmentation=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.train = train
        self.dspos = data_train_pos
        self.dsneg = data_train_neg
        self.augmentation= augmentation
    
    def __getitem__(self, index):
        text = str(self.ds[index][1])
        if self.train:
            label = self.ds[index][2]
                         
        if self.train:
            labels_tensors = torch.tensor(int(label))
            if self.augmentation:
                if labels_tensors > 0:
                    diagnoses_num = random.randint(0,10)
                    random_sequence = torch.randperm(len(self.dspos))
                    random_diagnoses_idx = random_sequence[:diagnoses_num]
                    random_diagnoses = self.dspos[random_diagnoses_idx,1]
                    if diagnoses_num>1:
                        for d in random_diagnoses:
                            text+=' '
                            text+=str(d)
                else:
                    diagnoses_num = random.randint(0,10)
                    random_sequence = torch.randperm(len(self.dsneg))
                    random_diagnoses_idx = random_sequence[:diagnoses_num]
                    random_diagnoses = self.dsneg[random_diagnoses_idx,1]
                    if diagnoses_num>1:
                        for d in random_diagnoses:
                            text+=' '
                            text+=str(d)                
        text = text[:1500]                
        text_token = self.tokenizer.encode(str(text), add_special_tokens=True)
                
        text_tensors = torch.tensor(text_token,dtype=torch.long)
        
        if len(text_tensors) > 512:
            text_tensors = text_tensors[:512]
            
        if self.train:
            sample = {'src_token':text_tensors,
                      'trg':labels_tensors,
                      }
        else:
            sample = {'src_token':text_tensors,
                      }            
        return sample
    
    def __len__(self):
        return self.len

class task1ds_electra(Dataset):
    def __init__(self,ds,tokenizer,train=True,data_train_pos=None):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.train = train
        self.dspos = data_train_pos
    
    def __getitem__(self, index):
        text = str(self.ds[index][1])
        if self.train:
            label = self.ds[index][2]
                         
        if self.train:
            labels_tensors = torch.tensor(int(label))
            if labels_tensors > 0:
                diagnoses_num = random.randint(3,10)
                random_sequence = torch.randperm(len(self.dspos))
                random_diagnoses_idx = random_sequence[:diagnoses_num]
                random_diagnoses = self.dspos[random_diagnoses_idx,1]
                if diagnoses_num>1:
                    for d in random_diagnoses:
                        text+=d
        text = text[:1500]                
        text_token = self.tokenizer.encode(str(text), add_special_tokens=True)
                
        text_tensors = torch.tensor(text_token,dtype=torch.long)
        
        if len(text_tensors) > 512:
            text_tensors = text_tensors[:500]
            
        if self.train:
            sample = {'src_token':text_tensors,
                      'trg':labels_tensors,
                      }
        else:
            sample = {'src_token':text_tensors,
                      }            
        return sample
    
    def __len__(self):
        return self.len

def collate_fn_electra(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    labels_tensors = [DD['trg'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True,
                                  padding_value=0)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
#        labels_tensors[i,e:] = -1

    batch['src_token'] = tokens_tensors
    batch['trg'] = torch.tensor(labels_tensors)
    batch['mask_padding'] = masks_tensors
    batch['origin_seq_length'] = torch.tensor(origin_seq_length)
    
    return batch


def collate_fn_electra_test(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]

    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True,
                                  padding_value=0)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
#        labels_tensors[i,e:] = -1

    batch['src_token'] = tokens_tensors
    batch['mask_padding'] = masks_tensors
    batch['origin_seq_length'] = torch.tensor(origin_seq_length)
    
    return batch

class task1ds(Dataset):
    def __init__(self,ds,tokenizer,clamp_size,train=True):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size
        self.train = train
    
    def __getitem__(self, index):
        text = self.ds[index][1]
        if self.train:
            label = self.ds[index][2]
        text_list = ['|']
        text_list += list(str(text))
                       
        text_token = self.tokenizer.tokenize(text_list)
        
#        indexed_tokens_inp = self.tokenizer.encode(st)
#        indexed_tokens_trg = self.tokenizer.encode(st)
        
        tokens_tensors = torch.tensor(text_token)
        if self.train:
            labels_tensors = torch.tensor(int(label))
        
        if self.train:
            sample = {'src_token':tokens_tensors,
                      'trg':labels_tensors,
                      }
        else:
            sample = {'src_token':tokens_tensors,
                      }            
        return sample
    
    def __len__(self):
        return self.len

def collate_fn_lstm(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    labels_tensors = [DD['trg'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
    
    labels_tensors = torch.tensor(labels_tensors)
    
    temp = torch.tensor(origin_seq_length)
    
    sorted_seq_length,sorted_seq = temp.sort(descending=True)
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
    
    batch['src_token'] = tokens_tensors[sorted_seq]
    batch['trg'] = labels_tensors[sorted_seq]
    batch['mask_padding'] = masks_tensors[sorted_seq]
    batch['origin_seq_length'] = sorted_seq_length

    return batch

def collate_fn_lstm_test(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
    
    temp = torch.tensor(origin_seq_length)
    
    sorted_seq_length,sorted_seq = temp.sort(descending=True)
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 

    batch['src_token'] = tokens_tensors[sorted_seq]
    batch['mask_padding'] = masks_tensors[sorted_seq]
    batch['origin_seq_length'] = sorted_seq_length

    return batch

def collate_fn_lstm_pretrain(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    labels_tensors = [DD['trg'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
       
    temp = torch.tensor(origin_seq_length)
    
    sorted_seq_length,sorted_seq = temp.sort(descending=True)
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True,
                                  padding_value=0)
    labels_tensors = pad_sequence(labels_tensors, 
                                  batch_first=True,
                                  padding_value=-1)

    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    for i, e in enumerate(origin_seq_length):
        masks_tensors[i,:e+1] = 1 
    
    batch['src_token'] = tokens_tensors[sorted_seq]
    batch['trg'] = labels_tensors[sorted_seq]
    batch['mask_padding'] = masks_tensors[sorted_seq]
    batch['origin_seq_length'] = sorted_seq_length

    return batch

def make_cloze(src,tokenize_alphabets,percent=0.15, fix=False):
    max_len = len(src)
    err_sequence = torch.randperm(max_len)
    err_replace = int(percent*max_len)
    err_sequence = err_sequence[:err_replace]

    numbers = torch.arange(tokenize_alphabets.alphabet2idx['0'],
                           tokenize_alphabets.alphabet2idx['9']+1)
    alphabets = torch.arange(tokenize_alphabets.alphabet2idx['A'],
                             tokenize_alphabets.alphabet2idx['z']+1)
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['^']]
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['`']]
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx['[']]
    alphabets = alphabets[alphabets!=tokenize_alphabets.alphabet2idx[']']]
    
    numbers_li = list(numbers.numpy())
    alphabets_li = list(alphabets.numpy())
    others_li = [o for o in range(tokenize_alphabets.n_alphabets) if o not in numbers_li+alphabets_li]
    others = torch.tensor(others_li)
    
    others = others[others!=tokenize_alphabets.alphabet2idx['$']]
    others = others[others!=tokenize_alphabets.alphabet2idx['|']]
    others = others[others!=tokenize_alphabets.alphabet2idx['!']]
    others = others[others!=tokenize_alphabets.alphabet2idx['#0#']]
    others = others[others!=tokenize_alphabets.alphabet2idx['#1#']]
    others = others[others!=tokenize_alphabets.alphabet2idx['#2#']]
    others = others[others!=tokenize_alphabets.alphabet2idx['#3#']]                                                        
    others = others[others!=tokenize_alphabets.alphabet2idx['#4#']]
    others = others[others!=tokenize_alphabets.alphabet2idx['#5#']]                                                        
    others = others[others!=tokenize_alphabets.alphabet2idx['#6#']]
                                                            
    s = src.clone()
    for e in err_sequence:
        if s[e] in alphabets:
            if fix:
                s[e] = tokenize_alphabets.alphabet2idx['^']
            else:
                r = random.random()
                if r>0.2:
                    s[e] = tokenize_alphabets.alphabet2idx['^']
                elif r <= 0.2 and r > 0.1:
#                    random_idx = random.randint(0, len(alphabets)-1)
#                    s[e] = alphabets[random_idx]
                    s[e] = random.choice(alphabets)
        elif s[e] in numbers:
            if fix:
                pass
#                s[e] = tokenize_alphabets.alphabet2idx['^']
            else:
                r = random.random()
                if r>0.8:
                    s[e] = tokenize_alphabets.alphabet2idx['^']
                elif r <= 0.8 and r > 0.4:
#                    random_idx = random.randint(0, len(numbers)-1)
#                    s[e] = numbers[random_idx] 
                    s[e] = random.choice(numbers)
        elif s[e] ==0:
            if fix:
                pass
#                s[e] = tokenize_alphabets.alphabet2idx['^']
            else:
                r = random.random()
                if r>0.9:
                    s[e] = tokenize_alphabets.alphabet2idx['^']
                elif r <= 0.9 and r > 0.8:
#                    random_idx = random.randint(0, len(others)-1)
#                    s[e] = others[random_idx]
                    s[e] = random.choice(others)
        else:
            if fix:
                pass
#                s[e] = tokenize_alphabets.alphabet2idx['^']
            else:
                r = random.random()
                if r>0.8:
                    s[e] = tokenize_alphabets.alphabet2idx['^']
                elif r <= 0.8 and r > 0.4:
#                    random_idx = random.randint(0, len(others)-1)
#                    s[e] = others[random_idx]   
                    s[e] = random.choice(others)
                    
    rev_err_sequence = [idx for idx in range(max_len) if idx not in err_sequence]
    t = src.clone()
    t[rev_err_sequence] = -1
    
    return s, t

class D_stage1(Dataset):
    def __init__(self,ds,tokenizer,clamp_size):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
        self.clamp_size = clamp_size

    def __getitem__(self, index):
        text = self.ds[index][1]
        text_list = ['|']
        text_list += list(str(text))
                       
        text_token = self.tokenizer.tokenize(text_list)
        
        tokens_tensors = torch.tensor(text_token)

        if len(tokens_tensors) > self.clamp_size:
            clamp = random.randint(0,len(tokens_tensors)-self.clamp_size)
            tokens_tensors = tokens_tensors[clamp:clamp+self.clamp_size]

        s,t = make_cloze(tokens_tensors,self.tokenizer,
                         percent=0.15)
        
        sample = {'src_token':s,
                  'trg':t,
#                  'src':diagnosis_list,
                  }        
        return sample
   
    def __len__(self):
        return self.len


if __name__ == '__main__':
    
    fn = './data/train.csv'
    datas = pd.read_csv(fn,sep=';')
    
    ds = task1ds()
    print(0)

    
