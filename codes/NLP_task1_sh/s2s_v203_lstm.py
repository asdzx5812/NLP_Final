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

import task1_dataset_v02
import dg_alphabets
#import bert_unet_001
from alphabert_utils import clean_up, split2words, rouge12l, make_statistics
from alphabert_utils import save_checkpoint, load_checkpoint
import gruModel

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = ' '

batch_size = 8
device = 'cuda'
parallel = False

#device_ids = list(range(rank * n, (rank + 1) * n))

fn = './data/train.csv'
datas = pd.read_csv(fn,sep=';',dtype=str)
datasGold0 = datas[datas.Gold == '0']
datasGold1 = datas[datas.Gold == '1']
newset = []
newset.append(datasGold0)
for i in range(13):
    newset.append(datasGold1)
datanew = pd.concat(newset)
data_train = np.array(datanew)

fn2 = './data/test.csv'
data2 = pd.read_csv(fn2,sep=';',encoding='utf8',dtype=str)
data_test = np.array(data2)

data_pretrain = pd.concat([datas[['Index','Text']],data2])
data_pretrain = np.array(data_pretrain)

all_expect_alphabets = [' ', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
 '/', ':', ';', '<', '=', '>', '?', '@', '0', '1', '2', '3', '4', '5', '6', '7', '8',  
 '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
 'y', 'z', '^', '~', '`', '$', '!', '[', ']', '{', '}', '|', '#0#', '#1#','#2#', '#3#',
 '#4#','#5#','#6#','#7#',]

unsame = 0 
same = 0
max_txt = 0

tokenize_alphabets = dg_alphabets.Diagnosis_alphabets()
for a in all_expect_alphabets:
    tokenize_alphabets.addalphabet(a)

config = {'hidden_size': 64,
          'max_position_embeddings':1350,
          'eps': 1e-7,
          'input_size': tokenize_alphabets.n_alphabets, 
          'vocab_size': tokenize_alphabets.n_alphabets, 
          'hidden_dropout_prob': 0.1,
          'num_attention_heads': 8, 
          'attention_probs_dropout_prob': 0.1,
          'intermediate_size': 256,
          'num_hidden_layers': 16,
          }

D2S_datatrain = task1_dataset_v02.task1ds(data_train,
                                          tokenize_alphabets,
                                          clamp_size=config['max_position_embeddings'],
                                          train=True)

D2S_trainloader = DataLoader(D2S_datatrain,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=task1_dataset_v02.collate_fn_lstm)


D2S_test = task1_dataset_v02.task1ds(data_test,
                                     tokenize_alphabets,
                                     clamp_size=config['max_position_embeddings'],
                                     train=False)

D2S_testloader = DataLoader(D2S_test,
                           batch_size=1,
                           shuffle=False,
                           num_workers=4,
                           collate_fn=task1_dataset_v02.collate_fn_lstm_test)


D2S_datapretrain = task1_dataset_v02.D_stage1(data_pretrain,
                                          tokenize_alphabets,
                                          clamp_size=config['max_position_embeddings'],
                                          )

D2S_prentrainloader = DataLoader(D2S_datapretrain,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=task1_dataset_v02.collate_fn_lstm_pretrain)
 
lstm_model = gruModel.LSTM_baseModel(config)
try:
    checkpoint_file = './checkpoint_lstm'
    lstm_model = load_checkpoint(checkpoint_file,'lstm_pretrain.pth',lstm_model)
except:
    print('*** No Pretrain_Model ***')
    pass

def test_lstm(DS_model,dloader):
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
            
            
            src = src.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            pred_prop,_ = DS_model(x=src,
                                 x_lengths=origin_len)
            
            pred_.append(pred_prop[:,0])
            if batch_idx % 100==0:
                print(batch_idx)
        pred = torch.cat(pred_,dim=0)
    return pred


def train_lstm(DS_model,dloader,lr=1e-4,epoch=100,log_interval=20,parallel=parallel):
    DS_model.to(device)
    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
    
    criterion = nn.CrossEntropyLoss().to(device)
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
            
            src = src.float().to(device)
            trg = trg.float().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
        
            pred_prop,_ = DS_model(x=src,
                                 x_lengths=origin_len)
            
#            print (pred_prop[:,0].shape, trg.shape)

            loss = criterion(pred_prop[:,0],trg.long())
            
            loss.backward()
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
                
            if iteration % log_interval == 0:
#                step_loss.backward()
#                model_optimizer.step()
#                print('+++ update +++')
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item()))
#                print(0,st_target)
#                step_loss = 0
                
            if iteration % 400 == 0:
                checkpoint_file = './checkpoint_lstm'
                save_checkpoint(checkpoint_file,'lstm_pretrain.pth',DS_model,model_optimizer,parallel)
                    
            iteration +=1
        if ep % 1 ==0:
            checkpoint_file = './checkpoint_lstm'
            save_checkpoint(checkpoint_file,'lstm_pretrain.pth',DS_model,model_optimizer,parallel)


            print('======= epoch:%i ========'%ep)
                  
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./total_loss_finetune.csv', sep = ',')
    print(total_loss)


def train_lstm_stage1(TS_model,dloader,lr=1e-4,epoch=10,log_interval=20,cloze_fix=True, parallel=True):
    global checkpoint_file
    TS_model.to(device)
    model_optimizer = optim.Adam(TS_model.parameters(), lr=lr)
    if parallel:
        TS_model = torch.nn.DataParallel(TS_model)  
    TS_model.train()
#    criterion = alphabert_loss.Alphabert_satge1_loss(device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    iteration = 0
    total_loss = []    
    out_pred_res = []
    out_pred_test = []
    for ep in range(epoch):
        t0 = time.time()
#        step_loss = 0
        epoch_loss = 0
        epoch_cases = 0
        for batch_idx, sample in enumerate(dloader):
#            TS_model.train()
            model_optimizer.zero_grad()
            loss = 0
            
            src = sample['src_token']
            trg = sample['trg']
            att_mask = sample['mask_padding']
            origin_len = sample['origin_seq_length']            
            bs, max_len = src.shape
        
            src = src.float().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
#            return src,trg,att_mask,origin_len
            _, prediction_scores = TS_model(x=src,x_lengths=origin_len)                   
            loss = criterion(prediction_scores.view(-1,100).contiguous(),trg.view(-1).contiguous())
            loss.backward()
            model_optimizer.step()
                        
            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs
            
                if iteration % log_interval == 0:
                    print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                            ep, batch_idx * batch_size,
                            100. * batch_idx / len(dloader),
                            (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                            loss.item()))
                    
                if iteration % 400 == 0:
                    checkpoint_file = './checkpoint_lstm'
                    save_checkpoint(checkpoint_file,'lstm_pretrain.pth',TS_model,model_optimizer,parallel)
                    a_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
                    print(a_)
                    print(' ******** ******** ******** ')
                    _, show_pred = torch.max(prediction_scores[0],dim = 1)
                    err_cloze_ = trg[0] > -1
                    src[0][err_cloze_] = show_pred[err_cloze_].float()
                    b_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
                    print(b_)
                    print(' ******** ******** ******** ')
                    src[0][err_cloze_] = trg[0][err_cloze_].float()
                    c_ = tokenize_alphabets.convert_idx2str(src[0][:origin_len[0]])
                    print(c_)
                    
                    out_pred_res.append((ep,a_,b_,c_,err_cloze_))
                    out_pd_res = pd.DataFrame(out_pred_res)
                    out_pd_res.to_csv('./out_pred_train.csv', sep=',')
                                
            iteration +=1
        if ep % 1 ==0:
            checkpoint_file = './checkpoint_lstm'
            save_checkpoint(checkpoint_file,'lstm_pretrain.pth',TS_model,model_optimizer,parallel)

            print('======= epoch:%i ========'%ep)
            
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./result/total_loss_pretrain.csv', sep = ',')
    print(total_loss)

if task == 'train':
    train_lstm(lstm_model,D2S_trainloader,lr=1e-5,epoch=200,log_interval=3,parallel=parallel)
elif task == 'pretrain':
    train_lstm_stage1(lstm_model,D2S_prentrainloader,lr=1e-5,epoch=100,log_interval=20,cloze_fix=True, parallel=parallel)
elif task == 'test':
    res = test_lstm(lstm_model,D2S_testloader)
    aa, ans =torch.max(res,dim=1)
    ans = ans.int()
    data2['Gold'] = ans.cpu().numpy()
    rrr = data2[['Index','Gold']]
    rrr.to_csv('./task1_method2.csv',header=True,index=None)
    
# test_alphaBert(lstm_model,D2S_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_cyy_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
# test_alphaBert(lstm_model,D2S_lin_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)
#test_alphaBert(lstm_model,D2S_all_testloader,threshold=0.38, is_clean_up=True, ep='f1_Max_lstm_test',mean_max='max',rouge=True)