#!/usr/bin/env python
# coding: utf-8

# In[221]:


import datetime
import pandas as pd
import pickle
import os

# 여태까지 작업한 데이터가 담기는 곳
dict_df = {}
cnt = 0
trial = str(cnt)+'_trial'

# 업로드한 데이터가 담기는곳
pkl_data = None

# Save 하는 data가 저장되는 path
directory = './modules/pkl_file/'





class data_store():
    
    def __init__(self,train_df = pd.DataFrame() ,test_df = pd.DataFrame() ,all_df = pd.DataFrame()):
        
        global dict_df
        global cnt
        global trial
        global pkl_data
        
        
        self.dict_df = dict_df
        self.cnt = cnt
        self.trial = trial
        self.pkl_data = pkl_data
        
        
        self.train_df = train_df
        self.test_df = test_df
        self.all_df = all_df
        
        if self.pkl_data == None or self.pkl_data == {}:
            # 현재 데이터 받기용
            print('No saved pickle files')
            
        
        else : 
            print('Load the saved pickle file')
            # load할 pkl파일이 있으면 해당 데이터 load
            self.dict_df = self.pkl_data
            self.cnt = int(list(self.pkl_data.keys())[-1].split('_')[0])
            self.trial = list(self.pkl_data.keys())[-1]

        
        # 데이터 history용 
        if not self.all_df.empty :
            self.cnt+=1
            self.trial = str(self.cnt)+'_trial'
            self.dict_df[self.trial] = self.all_df

            
            
        elif not self.train_df.empty and not self.test_df.empty :
            self.train_df['data_set'] = 'train'
            self.test_df['data_set'] = 'test'
            
            for i in self.train_df.columns.difference(self.test_df.columns):
                self.test_df[str(i)] = 0
            
            self.all_df = pd.concat([self.train_df,self.test_df])
            
            self.cnt+=1
            self.trial = str(self.cnt)+'_trial'
            self.dict_df[self.trial] = self.all_df
            
        elif (not self.train_df.empty and self.test_df.empty) or (self.train_df.empty and not self.test_df.empty):
            print("train_df, test_df 인자를 모두 넣어 주어야 해요")
        
    
    def add_df(self,train_df = pd.DataFrame() ,test_df = pd.DataFrame() ,all_df = pd.DataFrame()):

        
        self.train_df = train_df
        self.test_df = test_df
        self.all_df = all_df
        
        # 데이터 history용 
        if not self.all_df.empty :
            self.cnt+=1
            self.trial = str(self.cnt)+'_trial'
            self.dict_df[self.trial] = self.all_df
            
            
        elif not self.train_df.empty and not self.test_df.empty :
            self.train_df['data_set'] = 'train'
            self.test_df['data_set'] = 'test'
            
            for i in self.train_df.columns.difference(self.test_df.columns):
                self.test_df[str(i)] = 0
            
            self.all_df = pd.concat([self.train_df,self.test_df])
            
            self.cnt+=1
            self.trial = str(self.cnt)+'_trial'
            self.dict_df[self.trial] = self.all_df
            
            
        elif (not self.train_df.empty and self.test_df.empty) or (self.train_df.empty and not self.test_df.empty):
            print("train_df, test_df 인자를 모두 넣어 주어야 해요")
        
        else : 
            print("train_df,test_df / all_df 인자를 추가해 주어야 데이터가 추가 돼요")
                    
#     def return_train_df(self):
#         return self.train_df
        
#     def return_test_df(self):
#         return self.test_df
    
#     def return_all_df(self):
#         return self.all_df
    
    def return_dict_df(self):
        return self.dict_df
    
    def return_last_trial(self):
        return self.trial

    def save_pkl(self, df = pd.DataFrame()):
        if df.empty : 
            with open(directory + datetime.datetime.now().strftime('%Y%m%d_%H%M')+'.pkl', 'wb') as f:
                pickle.dump(self.dict_df, f)
        
        
        else : 
            with open(directory + datetime.datetime.now().strftime('%Y%m%d_%H%M')+'.pkl', 'wb') as f:
                    pickle.dump(df, f)
            
        
    
    def del_last_hist_df(self):
        if self.cnt == 0 :
            print("삭제할 데이터 없음")
            self.dict_df = dict_df
            self.cnt = 0
            self.trial = str(cnt)+'_trial'
            self.pkl_data = None

            self.train_df = pd.DataFrame()
            self.test_df = pd.DataFrame()
            self.all_df = pd.DataFrame()
            
            
            
        else : 
            del self.dict_df[self.trial]
            print('마지막 trial 데이터 제거')
            print('제거된 trial : ', self.trial)

            self.cnt -= 1
            self.trial = str(self.cnt)+'_trial'
        


# pkl파일 있으면 불러오는부분 
## 가장 최신 pkl만 불러옴
if not os.path.exists(directory):
    os.makedirs(directory)

r = [s for s in os.listdir(directory) if ".pkl" in s]

if len(r)>=1:
    print("There is data stored")
    with open(directory+r[-1], 'rb') as fr:
        pkl_data = pickle.load(fr)
    print("Last data fetch completed")

else : 
    print("There is not data stored")

