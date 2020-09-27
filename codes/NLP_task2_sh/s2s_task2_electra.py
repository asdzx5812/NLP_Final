import os
import time
import unicodedata
import random
import string
import re
import sys
import glob
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

import task2_dataset_v01 as task2_dataset
#import bert_unet_001
from alphabert_utils import clean_up, split2words, rouge12l, make_statistics
from alphabert_utils import save_checkpoint, load_checkpoint, convert_token2str
from transformers import ElectraTokenizer
import electraModel

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'val'

batch_size = 8
device = 'cuda'
parallel = False

#device_ids = list(range(rank * n, (rank + 1) * n))

def expand_pos_data(datasGold1):
    posdata = np.array(datasGold1,dtype='str')

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

D2S_datatrain = task2_dataset.task2_electra_train(files=files_train,
                                                  tokenizer=Electratokenizer,
                                                  train=True)

D2S_trainloader = DataLoader(D2S_datatrain,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=task2_dataset.collate_fn_electra)

fn = './data/val.csv'
data_val = pd.read_csv(fn,sep=';',dtype=object)

D2S_dataval= task2_dataset.task2_electra_test(ds=data_val,
                                              tokenizer=Electratokenizer)

D2S_valloader = DataLoader(D2S_dataval,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=task2_dataset.collate_fn_electra_test)

fn = './data/test.csv'
data_test = pd.read_csv(fn,sep=';',dtype=object)

data_test['Cause']=None
data_test['Effect']=None

D2S_datatest= task2_dataset.task2_electra_test(ds=data_test,
                                              tokenizer=Electratokenizer)

D2S_testloader = DataLoader(D2S_datatest,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=task2_dataset.collate_fn_electra_test)


try:
    checkpoint_file = './checkpoint_electra_task2'
    Electramodel = load_checkpoint(checkpoint_file,'electra_task2.pth',Electramodel)
except:
    print('*** No Pretrain_Model ***')
    pass
 

def test_electra(DS_model,dloader,tokenizer,ds):
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
            idx = sample['idx']
            bs = len(src)
#            print(batch_idx,src)            
            src = src.to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
                   
            outputs = DS_model(input_ids=src, 
                               attention_mask=att_mask)
                        
            for i in range(bs):
                pred = outputs[i].T                
                res = torch.argmax(pred[:,:origin_len[i]],dim=1)
                cause0 = tokenizer.convert_ids_to_tokens(src[i,res[0]:res[1]])
                Effect0 = tokenizer.convert_ids_to_tokens(src[i,res[2]:res[3]])
                cause1 = tokenizer.convert_ids_to_tokens(src[i,res[4]:res[5]])
                Effect1 = tokenizer.convert_ids_to_tokens(src[i,res[6]:res[7]])      
                
                idxs = ds['Index'].loc[idx[i].item()]                
                s = idxs.split('.')
                
                if len(s)<3:
                    # answer_c = ' '.join(cause0)
                    # answer_e = ' '.join(Effect0)
                    answer_c = convert_token2str(cause0)
                    answer_e = convert_token2str(Effect0)
                else:
                    if s[2]==1:
                        # answer_c = ' '.join(cause0)
                        # answer_e = ' '.join(Effect0)
                        answer_c = convert_token2str(cause0)
                        answer_e = convert_token2str(Effect0)
                    else:
                        # answer_c = ' '.join(cause1)
                        # answer_e = ' '.join(Effect1)                        
                        answer_c = convert_token2str(cause1)
                        answer_e = convert_token2str(Effect1)    
                        
                pred_.append([answer_c,answer_e])                        
                
            if batch_idx % 100==0:
                print(batch_idx)
#        pred = torch.cat(pred_,dim=0)
    return pred_


def train_electra(DS_model,dloader,lr=1e-4,epoch=100,log_interval=20,parallel=parallel):
    DS_model.to(device)
    DS_model.train()
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
            
            src = src.long().to(device)
            trg = trg.long().to(device)
            att_mask = att_mask.float().to(device)
            origin_len = origin_len.to(device)
            
            outputs = DS_model(input_ids=src, 
                               attention_mask=att_mask)
            
            for i in range(bs):
                pred = outputs[i].T
                loss += criterion(pred[:,:origin_len[i]],trg[i].long())
                        
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
                checkpoint_file = '../checkpoint_electra_task2'
                save_checkpoint(checkpoint_file,'electra_task2.pth',DS_model,model_optimizer,parallel)
                    
            iteration +=1
        if ep % 1 ==0:
            checkpoint_file = '../checkpoint_electra_task2'
            save_checkpoint(checkpoint_file,'electra_task2.pth',DS_model,model_optimizer,parallel)


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
                  epoch=30,
                  log_interval=20,
                  parallel=parallel)
elif task == 'val':
    ans = test_electra(DS_model=Electramodel,
                       dloader=D2S_valloader,
                       tokenizer=Electratokenizer,
                       ds=data_val)
    ans_pd = pd.DataFrame(ans,columns=['Cause', 'Effect'])
    
    submit = data_val[['Index', 'Text', 'Cause', 'Effect']]
    
    submit['Cause'] = ans_pd['Cause']
    submit['Effect'] = ans_pd['Effect']
    
    submit.to_csv('./data/val_submit.csv',index=None,sep=';',encoding='utf-8')

elif task == 'test':
    ans = test_electra(DS_model=Electramodel,
                       dloader=D2S_testloader,
                       tokenizer=Electratokenizer,
                       ds=data_test)
    ans_pd = pd.DataFrame(ans,columns=['Cause', 'Effect'])
    
    submit = data_test[['Index', 'Text', 'Cause', 'Effect']]
    
    submit['Cause'] = ans_pd['Cause']
    submit['Effect'] = ans_pd['Effect']
    
    submit.to_csv('./data/task2_method1.csv',index=None,sep=';',encoding='utf-8')

#    aa, ans =torch.max(res,dim=1)
#    ans = ans.int()
#    data2['Gold'] = ans.cpu().numpy()
#    rrr = data2[['Index','Gold']]
#    rrr.to_csv('./submission_electra.csv',header=True,index=None)
