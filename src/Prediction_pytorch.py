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

class Prediction():
    def __init__(self, names, model, window_size, matrices, labels, dir_out, behaviors, device, saving=False):
        
        self.names=names
        self.matrices=matrices
        self.labels=labels
        self.model=model
        self.window_size=window_size
        self.dir_out=dir_out
        self.behaviors=behaviors
        self.device=device

        self.saving=saving
        
    def cut_full_slice(self, matrix, label):
            
        margin=int(0.1*self.window_size)
        stride=self.window_size-2*margin
    
        size=len(matrix)//stride-1
        nb_input=matrix.shape[1]
    
        X= np.empty([size,self.window_size,nb_input])
        Y= np.empty([size,self.window_size],dtype=int)

        for i in range(size):
            t=i*stride
            X[i,:,:]=matrix[t:t+self.window_size ,:]
            Y[i]=label[t:t+self.window_size] 
                       
        return X, Y #non -> keras.utils.to_categorical(Y,len(CLASSES))


    def stick(self,Y):
    
        window_size=Y.shape[1]
        margin=int(0.1*window_size)
    
        res=[]
        res.append(Y[0,:window_size-margin])
    
        for i in range(1,len(Y)):
            res.append(Y[i,margin:-margin])
        
        return np.concatenate(res,axis=0)


    def get_all(self):

        all_pred={}
        all_true={}

        for i,name in enumerate(self.names):
            matrix=self.matrices[name]
            label=self.labels[name]
            print(name,matrix.shape,label.shape)

            X_cut,Y_cut=self.cut_full_slice(matrix,label)
            Y_stick=self.stick(Y_cut)
            all_true[name]=Y_stick
        
            X_cut_tensor = torch.tensor(X_cut, dtype=torch.float32, device=self.device)
            X_cut_tensor=X_cut_tensor.permute(0, 2, 1).to(self.device)

            #prediction
            self.model.to(self.device)
            with torch.no_grad():
                Y_cut_pred_proba=self.model(X_cut_tensor)
                Y_cut_pred_proba = Y_cut_pred_proba.cpu().numpy().transpose(0, 2, 1)
                Y_cut_pred = np.argmax(Y_cut_pred_proba, axis=2)
            
            Y_pred=self.stick(Y_cut_pred)
            Y_pred_proba=self.stick(Y_cut_pred_proba)

            all_pred[name]=Y_pred

            if self.saving: 
                np.save(self.dir_out+"/"+name[0:-4]+"=Y_pred",Y_pred)
            #np.save(self.dir_out+"/"+name[0:-4]+"=Y_pred_proba",Y_pred_proba)
            #np.save(self.dir_out+"/"+name[0:-4]+"=Y_true",Y_stick)

        return all_pred,all_true
    
    def plot_categorical_vector(self, ax,vector,cat_selected,cat_names,markersize=2,color=None,label=None):
    
    #listes des couleurs par défaut
    #colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
        for cat in cat_selected:
            ax.set_label("toto")
            here=(vector==cat)
            x_here=np.arange(len(vector))[here]
            y_here=cat*np.ones(len(vector))[here]
            line,=ax.plot(x_here, y_here,'o',markersize=markersize,color=color)
        
        
        line.set_label(label)   
        
        ax.set_yticks(range(len(cat_selected)))
        ax.tick_params(colors='coral')
        labels=[]
        for cat in cat_selected:        
            labels.append(str(cat)+":"+cat_names[cat])
        
        ax.set_yticklabels(labels)
    
    def plot_compa_Y(self, ax,y,y_pred,size_pred=1,size_true=3):
    
        nb_output=len(self.behaviors)
        class_names=[CLASS.name for CLASS in self.behaviors]


        self.plot_categorical_vector(ax,y,range(nb_output),class_names,markersize=size_true,color="blue",label="Y_true")
        self.plot_categorical_vector(ax,y_pred,range(nb_output),class_names,markersize=size_pred,color="red",label="Y_pred")

        ax.legend()
    
    def plot_compa(self, selected_names, all_true, all_pred, title,deb=0,fin=-1):
        #deb=0
        #fin=-1
        nb=len(selected_names)
        fig,axs=plt.subplots(nb,1,figsize=(12,2*nb),sharex=False)
        if nb==1:axs=[axs]
        
        for i,name in enumerate(selected_names):
            self.plot_compa_Y(axs[i],all_true[name][deb:fin],all_pred[name][deb:fin])   
            axs[i].set_title(name, color="coral", fontsize=16)
        
        fig.tight_layout()
        fig.savefig(Path(self.dir_out, title))
        
    def plot_confusion_matrix(self, y_true, y_pred, classes,
                          normalize, axe,
                          title,
                          cmap="jet",
                          precision=2,
                         ):
    
    
        y_true=y_true.astype(int)
        y_pred=y_pred.astype(int)
    
        np.set_printoptions(precision=precision)
    
    
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
    
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        print("Confusion matrix, without normalization:")
        print(cm)
    
        if normalize:
                if axe=="col" :
                    cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                elif axe=="row":
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    print("Normalized confusion matrix")
                    print(cm)

    
        
        fig, ax = plt.subplots(figsize=(12,12))
        ax.grid(False)
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.'+str(precision)+'f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "coral")
        fig.tight_layout()    
        fig.savefig(Path(self.dir_out,title))
        
        