# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:46:13 2023

@author: ljeantet
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import torch 
from pathlib import Path
import time
import gc
import datetime
import pandas as pd

import seaborn as sns
sns.set_theme(style="dark")

class Prediction_Guyane():
    def __init__(self, names, model, window_size, matrices, dir_out, class_names, device, batch_size, saving=False):
        
        self.names=names
        self.matrices=matrices
        self.model=model
        self.compiled_model = torch.compile(model, backend="eager")
        self.window_size=window_size
        self.dir_out=dir_out
        self.class_names=class_names
        self.device=device
        self.batch_size=batch_size

        self.saving=saving
        
    def cut_full_slice(self, matrix):
            
        margin=int(0.1*self.window_size)
        stride=self.window_size-2*margin
    
        size=(len(matrix)//stride)-1
        nb_input=matrix.shape[1]
    
        X= np.empty([size,self.window_size,nb_input])
        

        for i in range(size):
            t=i*stride
            X[i,:,:]=matrix[t:t+self.window_size ,:]

                       
        return X


    def stick(self,Y):
    
        window_size=Y.shape[1]
        margin=int(0.1*window_size)
    
        res=[]
        res.append(Y[0,:window_size-margin])
    
        for i in range(1,len(Y)):
            res.append(Y[i,margin:-margin])
        
        return np.concatenate(res,axis=0)


    def get_all(self):

        #clean gpu
        torch.cuda.empty_cache()
        gc.collect()

        all_pred={}
        

        for i,name in enumerate(self.names):
            matrix=self.matrices[name]
            
            print(name,matrix.shape)

            X_cut=self.cut_full_slice(matrix)
            X_cut_tensor = torch.tensor(X_cut, dtype=torch.float32, device=self.device)
            X_cut_tensor=X_cut_tensor.permute(0, 2, 1).to(self.device)
            
            del X_cut
            
            #prediction

            start_time_individual= time.time()
            self.model.to(self.device)
            #print("start predict")
            predictions = []
            with torch.no_grad():
                for i in range(0, X_cut_tensor.size(0), self.batch_size):
                    batch = X_cut_tensor[i:i+self.batch_size]
                    pred=self.compiled_model(batch)
                    predictions.append(pred)
            #print("--- %s seconds ---" % (time.time() - start_time_individual))  
            Y_cut_pred_proba = torch.cat(predictions, dim=0)       
            Y_cut_pred_proba = Y_cut_pred_proba.cpu().numpy().transpose(0, 2, 1)
            Y_cut_pred = np.argmax(Y_cut_pred_proba, axis=2)
            
            
                        
            Y_pred=self.stick(Y_cut_pred)
      

            all_pred[name]=Y_pred

            if self.saving: 
                np.save(Path(self.dir_out,f"{name}=Y_pred"),Y_pred)
            #np.save(self.dir_out+"/"+name[0:-4]+"=Y_pred_proba",Y_pred_proba)
            #np.save(self.dir_out+"/"+name[0:-4]+"=Y_true",Y_stick)

        del Y_cut_pred_proba,Y_cut_pred, Y_pred 
        #clean gpu
        torch.cuda.empty_cache()
        gc.collect()

        return all_pred 
    
    

    def clean_short_behaviors(self, y, min_length=20):
        y = np.array(y)
        cleaned = y.copy()

        # Find segment boundaries
        change_points = np.where(np.diff(y) != 0)[0] + 1
        segment_starts = np.concatenate(([0], change_points))
        segment_ends = np.concatenate((change_points, [len(y)]))

        for start, end in zip(segment_starts, segment_ends):
            segment_length = end - start
            if segment_length < min_length:
                prev_value = y[start - 1] if start > 0 else y[end]
                cleaned[start:end] = prev_value

        return cleaned

    def plot_histogram(self, Y_stick_pred, TV):
        counts = {}
        for i in range(len(self.class_names)):
            here=Y_stick_pred==i
            counts[i]=np.arange(len(Y_stick_pred))[here].shape[0]
            counts
        plt.clf()
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.title("Behaviors distribution", fontsize = 16)
        ytics=range(len(self.class_names))
        ytics_label=self.class_names
        plt.xticks(ytics,ytics_label);
        plt.bar(counts.keys(), counts.values(), width=0.95, edgecolor='k');
        plt.savefig(self.dir_out+"/"+TV+"_behaviour_frequence")
                
    def plot_budget_time(self, Y_stick_pred,TV):
    
        unique_pred, counts_pred = np.unique(Y_stick_pred, return_counts=True)
    
        beh_count=[0,0,0,0,0,0,0]
        for i in unique_pred :
            beh_count[i]=counts_pred[np.where(unique_pred==i)[0]][0]

        colors = ['darkred', 'darkgray', 'chartreuse', 'gold','darkorange','deeppink','navy']
        labels = ['Autre','Respiration','Alimentation','Nage inactive','Repos','Se gratte','Nage active']
        sizes = [(beh_count/np.sum(counts_pred))*100]
        #patches, texts = plt.pie(sizes, colors=colors, shadow=True, autopct='%1.1f%%',startangle=90)
        fig1, ax1 = plt.subplots(figsize=(9, 9))
        fig1.subplots_adjust(0,0,1,1)
        ax1.pie(sizes[0], labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=90)
        #patches, texts = ax1.pie(sizes, colors=colors, shadow=False, startangle=90)
        #plt.legend(patches, labels, loc="upper right", bbox_to_anchor=(1.4, 0.8),fontsize=24)
        ax1.axis('equal')
        #ax1.title('Budget Time Predicted', color="coral", fontsize=16)
        #fig1.show()
        fig1.savefig(self.dir_out+"/"+TV+"_behaviour_piechart") 

    def creating_rblt_fichier(self, Y_stick_pred,TV,nb,time_interval,date_start):
        
        behav=Y_stick_pred.copy()

        states=np.diff(behav)!=0
        fin=np.append(np.where(states)[0],behav.shape[0]-1)
        start=np.insert(np.where(states)[0], 0, 0)


        behaviour=[]
        for i in range(len(behav)):
            behaviour.append(self.class_names[behav[i]])
        
        df = pd.DataFrame({'Behaviour_ind': Y_stick_pred, 'Behaviour': behaviour})

        Date_depart=datetime.datetime.strptime(time_interval[0],"%Y-%m-%d %H:%M:%S")
        Date_depart
        dt_ref=Date_depart
        time=[]
        for i in range(len(Y_stick_pred)):
            time.append(dt_ref)
            dt_ref += datetime.timedelta(milliseconds=50)
        df['datetime']=pd.to_datetime(time,format='%Y.%m.%d %H:%M:%S.%f')

        rblt = pd.DataFrame({'Behaviour_ind': [df.iloc[x,0] for x in fin], 'Behavior': [df.iloc[x,1] for x in fin], 'datetime_deb':[df.iloc[x,2] for x in start],'datetime_fin':[df.iloc[x,2] for x in fin]})
        deb_s=[(rblt.iloc[x,2]-date_start).total_seconds() for x in range(rblt.shape[0])]
        fin_s=[(rblt.iloc[x,3]-date_start).total_seconds() for x in range(rblt.shape[0])]
        rblt['Start (s)']=deb_s
        rblt['Stop (s)']=fin_s
        rblt['Duration (s)']=[fin_s[x]-deb_s[x] for x in range(len(deb_s))]
        
        rblt.to_csv(self.dir_out+"/"+TV+"_behaviours_rblt.csv",sep=';',index=False)
        div=rblt.shape[0]//nb
        rblt_cut=rblt.iloc[0:div,:].copy()
        rblt_cut.to_csv(self.dir_out+"/"+TV+"_behaviours_d0_rblt.csv",sep=';',index=False)

        for i in range(1,nb):
            w=div*i
            v=div*(i+1)
            rblt_cut=rblt.iloc[w:v,:].copy()
            rblt_cut.to_csv(self.dir_out+"/"+TV+"_behaviours_d"+str(i)+"_rblt.csv",sep=';',index=False)
        
        return rblt