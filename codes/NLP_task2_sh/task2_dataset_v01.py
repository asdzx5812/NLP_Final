import os
import time
import unicodedata
import random
import string
import re
import sys
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from alphabert_utils import split2words
from transformers import ElectraTokenizer, ElectraForTokenClassification
import electraModel


def convert_bert_area(text_tokens,ans_tokens):
    ans_tokens_len = len(ans_tokens)
    sos_seq = []
    for i in range(1,ans_tokens_len-1):
        t_num = text_tokens.count(ans_tokens[i])
        if t_num ==1:
            sos = text_tokens.index(ans_tokens[i])-i+1
            eos = sos+ans_tokens_len-3
            return sos, eos
        elif t_num<1:
            pass
        else:
            tos = -1
            for j in range(t_num-1):
                tos = text_tokens.index(ans_tokens[i],tos+1)          
                sos_seq.append(tos-i+1)
    sos = pd.Series(sos_seq).mode().loc[0]
    eos = sos+ans_tokens_len-3
                        
    return sos, eos

class task2_electra_train(Dataset):
    def __init__(self,files,tokenizer,train=True):
        self.len = len(files)
        self.ds = files
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        fn = self.ds[index]
        df = pd.read_csv(fn,encoding='utf-8',sep='\t')
        text = str(df['Text'].loc[0])
        
        label = []     
        
        if len(df)<2:
            cause0 = text[int(df['Cause_Start'].loc[0]):int(df['Cause_End'].loc[0])]
            Effect0 = text[int(df['Effect_Start'].loc[0]):int(df['Effect_End'].loc[0])]
            cause1 = text[int(df['Cause_Start'].loc[0]):int(df['Cause_End'].loc[0])]
            Effect1 = text[int(df['Effect_Start'].loc[0]):int(df['Effect_End'].loc[0])]
        else:
            cause0 = text[int(df['Cause_Start'].loc[0]):int(df['Cause_End'].loc[0])]
            Effect0 = text[int(df['Effect_Start'].loc[0]):int(df['Effect_End'].loc[0])]
            cause1 = text[int(df['Cause_Start'].loc[1]):int(df['Cause_End'].loc[1])]
            Effect1 = text[int(df['Effect_Start'].loc[1]):int(df['Effect_End'].loc[1])]           
                          
               
        text = text[:1500]                
        text_token = self.tokenizer.encode(str(text), add_special_tokens=True)
 
        cause_token0 = self.tokenizer.encode(str(cause0), add_special_tokens=True)
        Effect_token0 = self.tokenizer.encode(str(Effect0), add_special_tokens=True)

        cause_token1 = self.tokenizer.encode(str(cause1), add_special_tokens=True)  
        Effect_token1 = self.tokenizer.encode(str(Effect1), add_special_tokens=True)  
        
        sos, eos = convert_bert_area(text_token,cause_token0)
        label.append(sos)
        label.append(eos)
        sos, eos = convert_bert_area(text_token,Effect_token0)
        label.append(sos)
        label.append(eos)
        sos, eos = convert_bert_area(text_token,cause_token1)
        label.append(sos)
        label.append(eos)
        sos, eos = convert_bert_area(text_token,Effect_token1)
        label.append(sos)
        label.append(eos)
               
        text_tensors = torch.tensor(text_token,dtype=torch.long)
        
        labels_tensors = torch.tensor(label,dtype=torch.long)
        
        if len(text_tensors) > 512:
            text_tensors = text_tensors[:512]
            
        sample = {'src_token':text_tensors,
                  'trg':labels_tensors,
                  }
         
        return sample
    
    def __len__(self):
        return self.len


class task2_electra_test(Dataset):
    def __init__(self,ds,tokenizer):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        sample = self.ds.loc[index]
        text = str(sample['Text'])
              
        text = text[:1500]                
        text_token = self.tokenizer.encode(str(text), add_special_tokens=True)
                
        text_tensors = torch.tensor(text_token,dtype=torch.long)
        
        if len(text_tensors) > 512:
            text_tensors = text_tensors[:512]
            
        sample = {'src_token':text_tensors,
                  'idx': index
                  }            
        return sample
    
    def __len__(self):
        return self.len

class task2_electra_val(Dataset):
    def __init__(self,ds,tokenizer):
        self.len = len(ds)
        self.ds = ds
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        sample = self.ds.loc[index]
        text = str(sample['Text'])
              
        text = text[:1500]                
        text_token = self.tokenizer.encode(str(text), add_special_tokens=True)
                
        text_tensors = torch.tensor(text_token,dtype=torch.long)
        
        if len(text_tensors) > 512:
            text_tensors = text_tensors[:512]
            
        sample = {'src_token':text_tensors,
                  'idx': index
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
    batch['trg'] = torch.stack(labels_tensors)
    batch['mask_padding'] = masks_tensors
    batch['origin_seq_length'] = torch.tensor(origin_seq_length)
    
    return batch


def collate_fn_electra_test(datas):
    from torch.nn.utils.rnn import pad_sequence
    batch = {}
    tokens_tensors = [DD['src_token'] for DD in datas]
    origin_seq_length = [len(d) for d in tokens_tensors]
    idx = [DD['idx'] for DD in datas]

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
    batch['idx'] = torch.tensor(idx)
    
    return batch


if __name__ == '__main__':
#    import pandas as pd
#    fn = './data/train.csv'
#    datas = pd.read_csv(fn,sep=';',dtype=object)
#    
#    data_val = datas.sample(frac=0.1,random_state=1)
#    data_train = datas.drop(data_val.index)
    
#    ds = task1ds_electra_train()
    

    fp = './data/each_train/'
    files_train = glob.glob(fp+'*.csv')
    
    
    fp = './data/each_val/'
    files_val = glob.glob(fp+'*.csv')
    
    unsame = 0 
    same = 0
    max_txt = 0
    
    pretrainfile ="google/electra-base-discriminator"
    Electratokenizer = ElectraTokenizer.from_pretrained(pretrainfile)
    Electramodel = electraModel.ElectraForQuestionAnswering(pretrainfile=pretrainfile)
    
    D2S_datatrain = task2_electra_train(files=files_train,
                                                      tokenizer=Electratokenizer,
                                                      train=True)    
    tt =D2S_datatrain[1]    
    print(0)

    
