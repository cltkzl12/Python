# 추상화 클래스 선언
from abc import *

# 모델 불러오기
import lightgbm as lgb

# config파일 불러오기
from py_code import config

#log.py 파일 불러오기
from py_code import log

# 기본 라이브러리 불러오기
import numpy as np
import pandas as pd
from datetime import datetime

class model(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,train,test,params,input_cols,trial):
        pass
    
    @abstractmethod
    def _custom_train_valid_split(self):
        pass
    
    @abstractmethod
    def _custom_performance():
        pass
    
    @abstractmethod
    def _train(self):
        #model.train(feval = _custom_performance)
        pass
 
    @abstractmethod
    def predict(self,mode=''):
        if mode == 'train':
            #train,valid = self._custom_train_Valid_split(train) or CV
            pass
        elif mode == 'test':
            pass
        else :
            return 0
    
    @abstractmethod
    def proba(self):
        pass

    

class lgb_model(model):
    def __init__(self,train,test,model_params = [],fit_params=[],input_cols = [],trial=''):
        self.model_params = model_params
        self.fit_params = fit_params
        
        self.input_cols = input_cols
        
        self.train = train
        self.test = test
        self.trial = trial
        
        print('input_cols :',self.input_cols)
        print('params :',self.model_params,self.fit_params)
    
    def _custom_train_valid_split(self,data,cutoff_day = 15):
        train = data[data['day'] <= cutoff_day]
        valid = data[data['day'] > cutoff_day]
        return train, valid
        
    def _custom_performance(self,y_pred, data):
        y_true = np.array(data.get_label())
        score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))
        return 'rmsle', score, False
    

    def _train(self,train_ds,val_ds):
        
        model = lgb.train(
                self.model_params,
                train_ds,
                **self.fit_params,
                valid_sets = [train_ds, val_ds],
                valid_names = config.valid_names,
                evals_result = config.evals_result,
                feval = self._custom_performance # <=============
            )
        return model

        
    def predict(self,mode = ''):
        
        def prep_train_data(data, input_cols):
            X = data[input_cols].values
            y_r = data['registered_log'].values
            y_c = data['casual_log'].values
            return X, y_r, y_c
        
        if mode == 'train':
            train, valid = self._custom_train_valid_split(self.train)

            X_train, y_train_r, y_train_c = prep_train_data(train, self.input_cols)
            X_valid, y_valid_r, y_valid_c = prep_train_data(valid, self.input_cols)

            train_r_ds = lgb.Dataset(X_train,y_train_r)
            val_r_ds = lgb.Dataset(X_valid,y_valid_r)

            train_c_ds = lgb.Dataset(X_train,y_train_c) 
            val_c_ds = lgb.Dataset(X_valid,y_valid_c)
            
            self.model_r = self._train(train_r_ds,val_r_ds)
            y_pred_r = np.exp(self.model_r.predict(X_valid)) - 1
            y_pred_t_r = np.exp(self.model_r.predict(X_train)) - 1
            
            self.model_c = self._train(train_c_ds,val_c_ds)
            y_pred_c = np.exp(self.model_c.predict(X_valid)) - 1
            y_pred_t_c = np.exp(self.model_c.predict(X_train)) - 1
            
            y_pred_comb = np.round(y_pred_r + y_pred_c)
            y_pred_comb[y_pred_comb < 0] = 0
            
            y_pred_t_comb = np.round(y_pred_t_r + y_pred_t_c)
            y_pred_t_comb[y_pred_t_comb < 0] = 0
            
            y_actual_t_comb = np.exp(y_train_r) + np.exp(y_train_c) - 2
            y_actual_comb = np.exp(y_valid_r) + np.exp(y_valid_c) - 2
            
            def get_rmsle(y_pred, y_actual):
                diff = np.log(y_pred + 1) - np.log(y_actual + 1)
                mean_error = np.square(diff).mean()
                return np.sqrt(mean_error)
            
            train_rmsle = get_rmsle(y_pred_t_comb,y_actual_t_comb)
            val_rmsle = get_rmsle(y_pred_comb, y_actual_comb)
            
            log.log_reg(self.trial, config.lgb_model_name, train_rmsle,val_rmsle)
            
            return (y_pred_comb, y_actual_comb, val_rmsle)
            
        elif mode == 'test':
            print('hi')
            
        else : 
            print('mode 설정이 필요 (train or test)')
        


    def proba(self):
        pass




