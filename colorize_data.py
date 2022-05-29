from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
import glob
from torchvision.transforms.functional import resize
from PIL import Image
from natsort import natsorted


class ColorizeData(Dataset):
    def __init__(self):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # self.input_transform = in_trans
        # self.target_transform = target_trans    
          
        #path = r"C:\Users\shalini\Downloads\train_landscape_images\landscape_images"
        #path = "C:\\Users\\shalini\\Downloads\\train_landscape_images"
        path = "/content/drive/MyDrive/image_color/image_color/landscape_images"
        #paths = glob.glob(path + "/*.jpg")
        self.paths = path
        print(path)
        all_imgs = os.listdir(path)
        self.total_imgs = natsorted(all_imgs)
    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.total_imgs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        #Return the input tensor and output tensor for training
        img_loc = os.path.join(self.paths, self.total_imgs[index])     
        #img_loc= self.paths + "/"+ str(index)+".jpg"
        #img_loc = os.path.join(self.path,img)
        #img_loc = "/content/drive/MyDrive/image_color/image_color/landscape_images/0.jpg"
        image = Image.open(img_loc).convert("RGB")
        input_val = self.input_transform(image)
        output = self.target_transform(image)

        return input_val,output
        
        