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
#import apex
#from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed

import task1_dataset_v03 as task1_dataset
import dg_alphabets
#import bert_unet_001
from alphabert_utils import clean_up, split2words, rouge12l, make_statistics
from alphabert_utils import save_checkpoint, load_checkpoint
from transformers import ElectraTokenizer
import electraModel

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test'

batch_size = 8
device = 'cuda'
parallel = False

#device_ids = list(range(rank * n, (rank + 1) * n))

def expand_pos_data(datasGold1):
    posdata = np.array(datasGold1,dtype='str')

fn = './data/train.csv'
datas = pd.read_csv(fn,sep=';',dtype=object)
datasGold0 = datas[datas.Gold == '0']
datasGold1 = datas[datas.Gold == '1']
newset = []
newset.append(datasGold0)
for i in range(13):
    newset.append(datasGold1)
datanew = pd.concat(newset)
data_train = np.array(datanew)
data_train_pos = np.array(datasGold1)
data_train_neg = np.array(datasGold0)


fn2 = './data/test.csv'
data2 = pd.read_csv(fn2,sep=';',encoding='utf8',dtype=object)
data_test = np.array(data2)

unsame = 0 
same = 0
max_txt = 0

pretrainfile ="google/electra-base-discriminator"
Electratokenizer = ElectraTokenizer.from_pretrained(pretrainfile)
Electramodel = electraModel.ElectraForTokenClassification(pretrainfile=pretrainfile)

D2S_datatrain = task1_dataset.task1ds_electra_train(data_train,
                                          Electratokenizer,
                                          train=True,
                                          data_train_pos=data_train_pos,
                                          data_train_neg=data_train_neg,
                                          augmentation=False)

D2S_trainloader = DataLoader(D2S_datatrain,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=task1_dataset.collate_fn_electra)


D2S_test = task1_dataset.task1ds_electra(data_test,
                                     Electratokenizer,
                                     train=False)

D2S_testloader = DataLoader(D2S_test,
                           batch_size=8,
                           shuffle=False,
                           num_workers=8,
                           collate_fn=task1_dataset.collate_fn_electra_test)
 

try:
    checkpoint_file = './checkpoint_electra'
    Electramodel = load_checkpoint(checkpoint_file,'electra_task1.pth',Electramodel)
except:
    print('*** No Pretrain_Model ***')
    pass

def test_electra(DS_model,dloader):
    DS_model.to(device)
    DS_model.eval()
    pred_ = []
    with torch.no_grad():
        
        t0 = time.time()
#        step_loss = 0
        for batch_idx, sample in enumerate(dloader):        
            
            src = sample['src_token']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
#            print(batch_idx,src)            
            src = src.to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
                   
            outputs = DS_model(input_ids=src, 
                               attention_mask=att_mask, 
                               )
            pred_.append(outputs[0])
            if batch_idx % 100==0:
                print(batch_idx)
        pred = torch.cat(pred_,dim=0)
    return pred


def train_electra(DS_model,dloader,lr=1e-4,epoch=100,log_interval=20,parallel=parallel):
    DS_model.to(device)
    DS_model.train()
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
    
#    criterion = nn.CrossEntropyLoss().to(device)
    iteration = 0
    total_loss = []
    for ep in range(epoch):
        
        t0 = time.time()
#        step_loss = 0
        epoch_loss = 0
        epoch_cases =0
        for batch_idx, sample in enumerate(dloader):  
            model_optimizer.zero_grad()
            loss = 0
            
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']
            
            bs = len(src)
            
            src = src.long().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            outputs = DS_model(input_ids=src, 
                               attention_mask=att_mask, 
                               labels=trg)
        
            loss, scores = outputs[:2]
                        
            loss.sum().backward()
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.sum().item()*bs
                epoch_cases += bs
                
            if iteration % log_interval == 0:
#                step_loss.backward()
#                model_optimizer.step()
#                print('+++ update +++')
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.sum().item()))
#                print(0,st_target)
#                step_loss = 0
                
            if iteration % 400 == 0:
                checkpoint_file = '../checkpoint_electra'
                save_checkpoint(checkpoint_file,'electra_task1.pth',DS_model,model_optimizer,parallel)
                    
            iteration +=1
        if ep % 1 ==0:
            checkpoint_file = '../checkpoint_electra'
            save_checkpoint(checkpoint_file,'electra_task1.pth',DS_model,model_optimizer,parallel)


            print('======= epoch:%i ========'%ep)
                  
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./total_loss_finetune_electra.csv', sep = ',')
    print(total_loss)


if task == 'train':
    train_electra(DS_model=Electramodel,
                  dloader=D2S_trainloader,
                  lr=1e-5,
                  epoch=3,
                  log_interval=20,
                  parallel=parallel)
elif task == 'test':
    res = test_electra(DS_model=Electramodel,
                 dloader=D2S_testloader)
    aa, ans =torch.max(res,dim=1)
    ans = ans.int()
    data2['Gold'] = ans.cpu().numpy()
    rrr = data2[['Index','Gold']]
    rrr.to_csv('./task1_method1.csv',header=True,index=None)
    
# test_alphaBert(lstm_model,D2S_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_cyy_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_lin_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
#test_alphaBert(lstm_model,D2S_all_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)