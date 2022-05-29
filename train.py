import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import colorize_data
import basic_model
import torch.nn as nn
## Optimizer used is ADAM and loss used is MSE loss or mean square error loss
## The model performance is measured using epoch wise and intermediate loss, that is continously added throughout an epoch.
## A constant learning rate is used for both the models
class Trainer:
    def __init__(self,lr,epoch=100):
        self.epoch = epoch
        self.learning_rate = lr
        dataset = colorize_data.ColorizeData()
        k = len(dataset)
        print(k)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(k*0.8),k-int(k*0.8)])
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
        # Define hparams here or load them from a config file
        self.model = basic_model.Net()
       
    def train(self):

        #model = basic_model.Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=0.0)
        size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        # train loop
      
        train_total_loss = 0
        correct = 0
      
        #train model
        #torch.cuda.empty_cache()
        i =0
        for batch,(data_in,data_out) in enumerate(self.train_dataloader):
            i = i + 1
            pred = self.model(data_in)
            loss = criterion(pred,data_out)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            #correct += (pred == data_out).float().sum()
            if batch % 1 == 0:
              loss, current = loss.item(), batch * len(data_in)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        #accuracy = 100 * correct /i
        #print("Train Accuracy:", accuracy)
        # Print Info Every Epoch
        print("Total Train Loss: ", train_total_loss)
        
    def validate(self):
      #model = Net()
      #model.eval()
      size = len(self.val_dataloader.dataset)
      num_batches = len(self.val_dataloader)
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=0.0)
      val_loss, correct_val = 0, 0
      #loss_best = 1e10
      i=0
      losses = 0 
      with torch.no_grad():
        for data_in,data_out in self.val_dataloader:
          i = i + 1
          output_val = self.model(data_in)
          losses += criterion(output_val, data_out).item()
          #correct_val += (output_val == data_out).float().sum()
     
      val_loss = losses
      val_loss /= num_batches
      #correct /= size
      #print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
      #accuracy = 100 * correct_val /i
      #print("Validation Accuracy:", accuracy)
      return val_loss,self.model, optimizer
        # Determine your evaluation metrics on the validation dataset.