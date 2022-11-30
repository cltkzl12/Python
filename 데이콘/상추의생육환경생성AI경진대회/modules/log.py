# 추상화 클래스 선언
from abc import *

# 기본 라이브러리
from datetime import datetime
import pandas as pd
import pickle
import os
import joblib

#config.py 불러오기
from py_code import config


class log(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,trial,model_name,train_eval,val_eval):
        pass
    
class log_reg(log):
    def __init__(self,trial,model_name,train_eval,val_eval ):
        self.cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.trial = trial
        self.model_name = model_name
        self.train_eval = train_eval
        self.val_eval = val_eval
        
        log_dic = {}
        log_dic['time'] = self.cur_time
        log_dic['trial'] = self.trial
        log_dic['model_name'] = self.model_name
        log_dic['train_eval'] = self.train_eval
        log_dic['val_eval'] = self.val_eval

        # log 파일이 없을때 생성
        path_dir = config.log_file_path
        file_list = os.listdir(path_dir)
        log_file_name = config.log_file_name
        
        if log_file_name not in file_list:
            log_pdf = pd.DataFrame(columns = ['time','trial','model_name','train_eval','val_eval'])
            joblib.dump(log_pdf,path_dir+log_file_name)
        
        
        # 해당 trial(data_name)/model_name가 같은 데이터중에서 val_eval이 낮을경우에만 log를 기록해 줄거임
        hist_log_pdf = joblib.load(path_dir+log_file_name)
        val_eval_list = list(hist_log_pdf[(hist_log_pdf['model_name']==self.model_name) & (hist_log_pdf['trial']==self.trial)]['val_eval'])
        
            
        if len(val_eval_list) == 0 :
            hist_log_pdf = hist_log_pdf.append(log_dic,ignore_index=True)
            joblib.dump(hist_log_pdf,path_dir+log_file_name)
            print('log save complete ; len = 0')
        else :
            min_val_eval = min(val_eval_list)
            if min_val_eval > self.val_eval:
                hist_log_pdf = hist_log_pdf.append(log_dic,ignore_index=True)
                joblib.dump(hist_log_pdf,path_dir+log_file_name)
                print('log save complete')
            else : 
                print('현재 로그가 더 높습니다.')
        
        
