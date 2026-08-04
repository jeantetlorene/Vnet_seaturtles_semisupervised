# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:59:51 2023

@author: ljeantet
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

class Global_generator(IterableDataset):

    def __init__(self, names, Behaviors, window_size, matrices, labels, descriptor_select, size_per_epoch):

        self.names=names
        self.window_size=window_size
        self.matrices=matrices
        self.labels=labels
        self.proba_of_classes=np.zeros(len(Behaviors))
        self.Behaviors=Behaviors
        self.descriptor_select=descriptor_select
        self.size_per_epoch = size_per_epoch

        for CLASS in self.Behaviors:
            self.proba_of_classes[CLASS.id]=CLASS.sampling_weight

        self.proba_of_classes/=np.sum(self.proba_of_classes)


        print([(CLASS.name,self.proba_of_classes[CLASS.id]) for CLASS in self.Behaviors])


        self.class_2_index_gene={}

        for CLASS in self.Behaviors:
        
            random_index_generators={}
            for name in names:
                _,_,_,gene=self.density_for_one_promotion(name, CLASS)
                random_index_generators[name]=gene

            self.class_2_index_gene[CLASS.name]=random_index_generators

    
    def rand_gene(self, cdf):
        while True:
            yield np.argmax(cdf>np.random.random())-self.window_size//2

    def density_for_one_promotion(self, ind_name, behav):

        label=self.labels[ind_name]

        N = len(label)
        dens = np.zeros_like(label)

        dens[label==behav.id] = 1

        sum_dens=np.sum(dens)
        if  sum_dens<1e-6:
            print("the class to promote:"+behav.name +" is not present in:"+ind_name)
            dens[:]=1
            sum_dens=len(dens)
        

        dens/=sum_dens

        # creating a mask. 
        kernel = np.ones(int(1.5*self.window_size))
        kernel/=np.sum(kernel)
        #gauss = np.exp(-t**2)
        #gauss /= np.sum(gauss)

        # convolving label with a mask
        dens_conv = np.convolve(dens, kernel, mode='same')
    
        dens_conv[-self.window_size//2-2:]=0
        dens_conv[:self.window_size//2+2]=0

        cdf=np.cumsum(dens_conv)
        cdf/=cdf[-1]

            
        return dens,dens_conv,cdf,self.rand_gene(cdf)
    
    def give_me_individual_and_timeIndex(self):

        if np.random.rand()<0.1:
            Ind=np.random.choice(self.names)
            deb=0
            fin=len(self.labels[Ind])-self.window_size
            t=np.random.randint(deb,fin)
            return Ind,t

        theClass=np.random.choice(self.Behaviors,p=self.proba_of_classes)
        Ind=np.random.choice(self.names)

    
        theGenerators=self.class_2_index_gene[theClass.name]
        theGenerator=theGenerators[Ind]
        
        return Ind, next(theGenerator)
    
    def __iter__(self):
        """Yields one epoch worth of balanced samples."""
        for i in range(self.size_per_epoch):
            individual_name, t = self.give_me_individual_and_timeIndex()
            matrix = self.matrices[individual_name]
            label = self.labels[individual_name]

            X = matrix[t:t+self.window_size, :]
            Y = label[t:t+self.window_size]
            Y_one_hot=F.one_hot(torch.tensor(Y, dtype=torch.long), num_classes=7)
            # Yield directly as PyTorch tensors
            yield (
                torch.from_numpy(X).float().permute(1,0),
                Y_one_hot.permute(1,0)
            )




    """
    def oneEpoch_generator_window(self, names, size=100):
       
        while True:
      
            X = np.empty([size,self.window_size,len(self.descriptor_select)])
            Y = np.empty([size,self.window_size],dtype=int)

            for i in range(size):
                individual_name,t=self.give_me_individual_and_timeIndex(names)
                matrix=self.matrices[individual_name]
                label=self.labels[individual_name]                        
                X[i,:,:]=matrix[t:t+self.window_size ,:]
                Y[i,:]=label[t:t+self.window_size]

            yield X,Y
    """
    