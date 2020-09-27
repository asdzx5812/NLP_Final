import random
import pandas as pd
import os

fn = './train.csv'
datas = pd.read_csv(fn,sep=';',dtype=object)
datas_ori = pd.read_csv(fn,sep=';',dtype=object)

file_train = './each_train/'
file_val = './each_val/'

idxs = datas['Index']

for i, idx in enumerate(idxs):
    s = idx.split('.')
    search_idx = ''.join([s[0],'.',s[1]])
    datas.loc[i]['Index']=search_idx

train_set = []
val_set = []

searched = []

idxs = datas['Index']    
for i, idx in enumerate(idxs):
    s = idx.split('.')
    search_idx = ''.join([s[0],'.',s[1]])
    if search_idx not in searched:
        searched.append(search_idx)
        each = datas[datas['Index']==search_idx]
        if random.random() >0.07:
            fn = os.path.join(file_train,str(s[0])+str(s[1])+'.csv')
            train_set.append(datas_ori[datas['Index']==search_idx])
        else:
            fn = os.path.join(file_val,str(s[0])+str(s[1])+'.csv')        
            val_set.append(datas_ori[datas['Index']==search_idx])
        each.to_csv(fn,index=None,sep='\t',encoding='utf-8')
    
val_df = pd.concat(val_set)
train_df = pd.concat(train_set)

val_df.to_csv('val.csv',index=None,sep=';',encoding='utf-8')
train_df.to_csv('tra.csv',index=None,sep=';',encoding='utf-8')