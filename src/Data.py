


import os
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import datetime


class classCLASS:

    def __init__(self,id,name,behaviors,sampling_weight=1,loss_weight=1):
        self.id=id
        self.name=name
        self.behaviors=behaviors
        self.sampling_weight=sampling_weight
        self.loss_weight=loss_weight

class Data: 
    def __init__(self, folder_in, folder_out, descriptor_select):
        self.dir_in=folder_in
        self.dir_out=folder_out
        self.descriptor_select=descriptor_select    
    
    def load_data_martinique(self, names, categories, beh_class, freq, verbose=False):

        matrices={}
        labels={}

        for name in names:
            full_name=self.dir_in+"/"+name+".npy"
            if not os.path.exists(full_name):
                assert 1==0,"the file:"+full_name+" cannot be found"
            else:
                full_mat=np.load(full_name, allow_pickle=True)
                pre_label=full_mat[:, beh_class]
            
                label=np.zeros_like(pre_label)
                for CLASS in categories:
                    if CLASS.name!="Other":
                        for beh in CLASS.behaviors:
                            label[pre_label==beh]=CLASS.id

                #np.save(self.dir_out+"/"+name[0:-3]+"=label",label)
                mat=full_mat[:,self.descriptor_select]
                matrices[name]=mat
                labels[name]=label
                if verbose :
                    print("loading the matrix:"+name)
                    print("shape:",mat.shape)
                    print("duration in seconds:",len(mat)/freq)
                    print("duration in hour:",round(len(mat)/freq/60/60,2))
                    print()
                
            
        return matrices,labels
    
    def load_data_guyane(self, names, freq, verbose=False):

        matrices={}
        for name in names:
                full_name=self.dir_in+"/"+name+".npy"
                if not os.path.exists(full_name):
                    assert 1==0,"le fichier:"+full_name+" cannot be found"
                else:
                    full_mat=np.load(full_name, allow_pickle=True)
                    mat=full_mat[:,self.descriptor_select]
                    matrices[name]=mat
                    if verbose : 
                        print("loading the matrix:"+name)
                        print("shape:",mat.shape)
                        print("duration in seconds:",len(mat)/freq)
                        print("duration in hour:",round(len(mat)/freq/60/60,2))
                        print()
                     
                                      
     
        del full_mat, mat   
        return matrices
        
        
    def load_dico(self, key_int=False, print_me=False):
        with open(self.dir_in+"/dico_info.json") as f:
                dico_str = json.loads(json.load(f))
    
   
        if key_int: 
            conv_key=lambda k:int(k)
        else:
            conv_key=lambda k:k
        
        dico={conv_key(k):v for k,v in dico_str.items()}
    
        if print_me:
            print(dico)
        
        return dico  
    
    def print_selected_variables(self, dico):
        
        va_names=dico['col_names']
        va_names_sel=[]
        for i,name in enumerate(va_names):
            if i in self.descriptor_select:
                prefix=""
                va_names_sel.append(name)
            else:
                prefix="-----suppressed----->"
            print(prefix,i,name)
        
        
    
    def create_saving_folder(self, Clear=False):
   
        if os.path.exists(self.dir_out) & Clear:
            shutil.rmtree(self.dir_out)
            print("we clear the directory:",self.dir_out)
            """creation of the folder """
            os.makedirs(self.dir_out)
        else:
            print("we create the directory or the directory already exists:",self.dir_out)
            os.makedirs(self.dir_out, exist_ok=True)
        
        
    def normalize(self, matrices):

        nb_col=len(self.descriptor_select)
        for matrix in matrices.values():
            for j in range(nb_col):
                matrix[:,j]-=matrix[:,j].mean()
                matrix[:,j]/=matrix[:,j].std()
        
        return matrices
    
    
    def histoOneIndividual(self, matrix):
        nb_col=matrix.shape[1]
        fig,axs=plt.subplots(1,nb_col,figsize=(nb_col*3,2))
        for j in range(nb_col):
            axs[j].hist(matrix[:,j],bins=40,edgecolor='k')
            axs[j].set_title(self.descriptor_select[j], color="coral")
            axs[j].tick_params(colors='coral')
        plt.show()



def get_start_and_time_interval(TV):
        if TV=='AGM1_TV_GF_21032022':
            KEPT_TIME_INTERVAL=("2022-03-22 02:01:22",None)
            START=datetime.datetime.strptime("2022-03-21 23:52:37","%Y-%m-%d %H:%M:%S")  
        elif TV=='AGM10_TV_GF_22032022':
            KEPT_TIME_INTERVAL=("2022-03-22 02:45:11","2022-04-03 06:30:58") 
            START=datetime.datetime.strptime("2022-03-22 01:36:59","%Y-%m-%d %H:%M:%S")
        elif TV=='AGM11_TV_GF_23032022':
            KEPT_TIME_INTERVAL=("2022-03-23 04:11:18",None) 
            START=datetime.datetime.strptime("2022-03-23 00:53:56","%Y-%m-%d %H:%M:%S")
        elif TV=='AGM12_TV_GF_23032022':
            KEPT_TIME_INTERVAL=("2022-03-23 05:45:57","2022-04-04 23:04:54")
            START=datetime.datetime.strptime("2022-03-23 00:41:12","%Y-%m-%d %H:%M:%S")  
        elif TV=='AGM10_TV_GF_18042023':
            KEPT_TIME_INTERVAL=("2023-04-18 02:07:06",None)
            START=datetime.datetime.strptime("2023-04-18 01:15:01","%Y-%m-%d %H:%M:%S")   
        elif TV=='AGM12_TV_GF_18042023':
            KEPT_TIME_INTERVAL=("2023-04-18 02:53:29",None)
            START=datetime.datetime.strptime("2023-04-18 01:38:01","%Y-%m-%d %H:%M:%S")
        elif TV=='CC_09_107_TV_GF_18042019':
            KEPT_TIME_INTERVAL=("2019-04-18 03:23:27", "2019-05-01 19:00:00")
            START=datetime.datetime.strptime("2019-04-18 02:07:49","%Y-%m-%d %H:%M:%S")
        return START, KEPT_TIME_INTERVAL       