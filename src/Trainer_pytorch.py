# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:23:46 2023

@author: ljeantet
"""

import os
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
   


class Trainer:
    def __init__(self, model, folder_out, optimizer, criterion, batch_size, loss_weight, device):
        
        self.model = model
        self.folder_out=folder_out
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size=batch_size
        self.loss_weight=loss_weight
        self.device = device
        self.best_loss = float('inf')
        self.total_duration = 0
        self.durations = []
    
    def train(self, train_dataset, val_dataset, epochs, patience=10, verbose=False):
        counter = 0
        self.train_losses=[]
        self.val_losses=[]
        self.train_accuracies, self.val_accuracies = [], []
        self.best_epochs=[]
        
        # Create DataLoaders ONCE
        X, Y = list(zip(*train_dataset))  # materialize generator into list
        X_tensor = torch.stack(X)
        Y_tensor = torch.stack(Y)
        
        training_dataset = TensorDataset(X_tensor, Y_tensor)
        train_loader = DataLoader(training_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        del X, Y, X_tensor, Y_tensor, training_dataset, train_dataset
        
        X, Y = list(zip(*val_dataset))  # materialize generator into list
        X_tensor = torch.stack(X)
        Y_tensor = torch.stack(Y)
        validation_dataset = TensorDataset(X_tensor, Y_tensor)
        val_loader = DataLoader(validation_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        del X, Y, X_tensor, Y_tensor, validation_dataset, val_dataset
        

        for epoch in tqdm(range(epochs)):

            starting_time = time.time()

            self.model.train(True)
            total_loss, total_acc = 0, 0
            
            batch_count = 0
            for x_batch, y_batch in train_loader:
                batch_count+=1
                x_batch = x_batch.to(self.device,non_blocking=True)
                y_batch = y_batch.to(self.device,non_blocking=True)
                # Reset gradients
                self.optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(x_batch)
                # Compute loss
                loss = self.criterion(y_pred, y_batch,self.loss_weight)
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                # Calculate Loss
                total_loss += loss.item()
                total_acc += self._compute_accuracy(y_pred, y_batch) 

            train_epoch_loss=total_loss/batch_count
            self.train_losses.append(train_epoch_loss)

            train_epoch_acc = total_acc / batch_count
            self.train_accuracies.append(train_epoch_acc)

            #evaluation
            val_loss, val_acc  = self._val_one_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if verbose : 
                print(f"Epoch {epoch+1}, Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            #early_stopping
            if val_loss < self.best_loss:
                
                self.best_loss = val_loss  
                self.best_epochs.append(epoch)  
                #save
                save_path = os.path.join(Path(self.folder_out, "model.pth"))
                torch.save(self.model.state_dict(), save_path)
                counter = 0

                if verbose :
                    print("val_loss improved from {:.5f} to {:.5f}".format(self.best_loss, val_loss))
                    print("Model is saved on epoch {:d}.".format(epoch))
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping due to no improvement")
                    break

            # saving training epoch history
            duration=time.time()-starting_time
            self.durations.append(duration)
            self.total_duration+=duration

    
    def _create_pytorch_dataloader(self, X, Y, shuffle):

        X_tensor = torch.from_numpy(X).float()
        X_tensor=X_tensor.permute(0, 2, 1)
        Y_tensor=Y.permute(0, 2, 1)
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=8,pin_memory=True
        )
        return loader
    
    
    def _compute_accuracy(self,y_pred, y_true):

        y_pred_labels = y_pred.argmax(dim=1)  # Shape: (batch_size, seq_length)
        y_true_labels = y_true.argmax(dim=1)  # Convert one-hot to class index (batch_size, seq_length)

        correct = (y_pred_labels == y_true_labels).float()
        return correct.mean().item()  # Return scalar accuracy


    def _val_one_epoch(self, val_loader):
        self.model.train(False)

        
        
        
        with torch.no_grad():
            total_loss, total_acc = 0.0, 0.0
            batch_count=0
            for x_batch, y_batch in val_loader:
                batch_count+=1
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                #forward pass
                y_pred = self.model(x_batch)
                #compute loss
                loss = self.criterion(y_pred, y_batch, self.loss_weight)
                #compute loss
                total_loss += loss.item()
                total_acc += self._compute_accuracy(y_pred, y_batch)

        return total_loss / batch_count, total_acc / batch_count
    
    def plot_train_val_loss(self, save=False):

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot( self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        if save: 
            plt.savefig(Path(self.folder_out, "Training and Validation Loss"))
        plt.show()


    def plot_train_val_acc(self, save=False):

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='Training Loss')
        plt.plot( self.val_accuracies, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        if save: 
            plt.savefig(Path(self.folder_out,"Training and Validation Accuracy"))

        plt.show()