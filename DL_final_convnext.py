#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# In[2]:


import pathlib
Dataset_Path = 'data_path/'
data_dir = pathlib.Path(Dataset_Path)


# In[3]:


import os
import numpy as np
import glob
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def GetFileName(root,root_len):
    filenames = glob.glob(os.path.join(root, '*_image1.png'))
    #check if is nan
    label = pd.read_csv(root+ "/train.csv")
    label = label.dropna()
    label = label['id']
    label = label.array
    #delete nan hand
    count = len(filenames) - 1
    while (count >= 0):
        if (filenames[count][-(len(filenames[count]))+root_len+1:-11] not in label):
            filenames.remove(filenames[count])
        count = count - 1
    return filenames

def replication (paths,root_len):
    label = pd.read_csv('data_path/train.csv',index_col='id')
    orilen = len(paths)
    for i in range(orilen):
        if label.loc[paths[i][-(len(paths[i]))+root_len+1:-11]][0] == 1:
            paths.append(paths[i])
    return paths

class GetDataSet(Dataset):
    def __init__(self,file,file_paths,train=True):
        self.file_paths = file_paths #file names of images
        self.train = train
        self.resize = transforms.Resize((224,224))
        if (self.train):
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(180, expand=True),
                transforms.Resize((224,224)),
                transforms.RandomEqualize(1),
                transforms.ToTensor()
                ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomEqualize(1),
                transforms.ToTensor()
                ])
        self.num_samples = len(self.file_paths)
        self.filenames = glob.glob(os.path.join(os.path.join('data_path/',file), "*.png"))
        self.label = pd.read_csv('data_path/train.csv',index_col='id')
        self.train_folder = 'data_path/train'

    def __getitem__(self,idx):
        file_path = self.file_paths[idx]
        img1 = torchvision.io.read_image(file_path,torchvision.io.ImageReadMode(1))
        img1 = self.transforms(img1)
        img1 = self.padding(img1)
        check = 0
        if(file_path[-len(file_path):-5]+'2.png' in self.filenames):
            img2 = torchvision.io.read_image(file_path[-len(file_path):-5]+'2.png',torchvision.io.ImageReadMode(1))
            img2 = self.transforms(img2)
            img2 = self.padding(img2)
            check = check + 1
        if(file_path[-len(file_path):-5]+'3.png' in self.filenames):
            img3 = torchvision.io.read_image(file_path[-len(file_path):-5]+'3.png',torchvision.io.ImageReadMode(1))
            img3 = self.transforms(img3)
            img3 = self.padding(img3)
            check = check + 1
        if(file_path[-len(file_path):-5]+'4.png' in self.filenames):
            img4 = torchvision.io.read_image(file_path[-len(file_path):-5]+'4.png',torchvision.io.ImageReadMode(1))
            img4 = self.transforms(img4)
            img4 = self.padding(img4)
            check = check + 1
        if (check == 3):img = torch.cat((img1,img2,img3,img4))
        elif (check == 2):img = torch.cat((img1,img2,img3,img1))
        elif (check == 1):img = torch.cat((img1,img2,img1,img1))
        else:img = torch.cat((img1,img1,img1,img1))
        
        if (self.train):
            return img,self.label.loc[file_path[-len(file_path)+len(self.train_folder)+1:-11]][0]
        else:
            return img
        
    def __len__(self):
        return self.num_samples

    def padding(self,img):
        h = img.size()[0]
        w = img.size()[1]
        if w != 224 or h != 224:
            if h > w:
                img = F.pad(img,((h-w)/2),((h-w)/2),0,0)
            elif w < h:
                img = F.pad(img,(0,0,(w-h)/2),((w-h)/2),0,0)
            img = self.resize(img)
        return img


# In[4]:


train_folder = 'data_path/train'
train_paths = GetFileName('data_path/train',len(train_folder))
train_paths = replication(train_paths,len(train_folder))
test_paths = glob.glob(os.path.join('data_path/test', '*_image1.png'))
train_dataset = GetDataSet('train',train_paths,train = True)
test_dataset = GetDataSet('test',test_paths,train = False)


# In[5]:


print(len(train_dataset))
print(len(test_dataset))


# In[6]:


#check dataset
#the first index stands for the index of the images
#the second index implies it is the image not the label
#the third is the index of the channel
img = train_dataset[1][0][0] 
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

img = train_dataset[1][0][1]
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

img = train_dataset[1][0][2]
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()

img = train_dataset[1][0][3]
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()


# In[7]:


#construct data loader
import torch.utils.data as data
batch_size = 16
VAL_RATIO = 0.2
percent = int(len(train_dataset) * (1 - VAL_RATIO))
train_set, valid_set = data.random_split(train_dataset, [percent, len(train_dataset)-percent])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# In[8]:


# Find the device available to use using torch library
device = torch.device("cuda")
#device = "cpu"


# In[9]:


from torch.autograd import Variable
model = torch.hub.load('pytorch/vision:v0.14.0', 'convnext_base', pretrained=True)
#turn off training for their parameters
for param in model.parameters():
    param.requires_grad = True
    
model.to(device)


# In[10]:


# Assuming `model` is your existing model
weight1 = getattr(model.features, '0')
model1 = getattr(weight1, '0')
weight1 = getattr(weight1, '0').weight.clone()

# Create a new first layer with 4 input channels
new_first_layer = nn.Conv2d(4, 128, kernel_size=(4, 4), stride=(4, 4))

# Copy the weights from the original first layer to the new first layer
new_first_layer.weight[:, :3, :, :].data = weight1

# Replace the original first layer in the model with the new first layer
setattr(model.features, '0', new_first_layer)

# Move the model to the desired device (e.g., GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# In[11]:


#create new classifier for model using torch.nn as nn library
#classifier_input = model.classifier.linear
classifier_input = getattr(model.classifier, '2')
classifier_input = classifier_input.in_features
num_labels = 2 #PUT IN THE NUMBER OF LABELS IN YOUR DATA
classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 256),
                           nn.ReLU(),
                           nn.Linear(256, num_labels),
                           nn.LogSoftmax(dim=1))
#replace default classifier with new classifier
setattr(model.classifier, '2', classifier)


# In[12]:


model.to(device)


# In[13]:


#set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
#set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(),lr = 0.0005)


# In[14]:


import matplotlib.pyplot as plt
#training
epochs = 50
best_valid_loss = 10
best_epoch = 0

train_loss_list = []
valid_loss_list = []
acc_train_list = []
acc_valid_list = []

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.type(torch.ByteTensor).to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        '''print(output,labels)'''
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)
        
        # Since our model outputs a LogSoftmax, find the real 
        # percentages by reversing the log function
        output = torch.exp(output)
        # Get the top class of the output
        top_p, top_class = output.topk(1, dim=1)
        # See how many of the classes were correct?
        equals = top_class == labels.view(*top_class.shape)
        # Calculate the mean (get the accuracy for this batch)
        # and add it to the running accuracy for this epoch
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
        # Print the progress of our training
        counter += 1
        print(counter, "/", len(train_loader), "Loss = ", loss)
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.type(torch.ByteTensor).to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            print("top_class: ", top_class)
            print("labels.view(*top_class.shape): ", labels.view(*top_class.shape))
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(val_loader), "Accuracy = ", val_accuracy)
    
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    train_acc = train_accuracy / len(train_loader)
    val_acc = val_accuracy / len(val_loader)
    
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    acc_train_list.append(train_acc)
    acc_valid_list.append(val_acc)
    
    #accuracy_list.append(accuracy/len(val_loader))
    if (valid_loss<best_valid_loss):
      best_valid_loss = valid_loss
      best_epoch = epoch
      #torch.save(model.state_dict(), '/content/drive/My Drive/Colab Notebooks/checkpoint.pth')
      print('The saved model is the {} epoch'.format(epoch))
    # Print out the information
    print('Train Accuracy: ', train_accuracy/len(train_loader))
    print('Validation Accuracy: ', val_accuracy/len(val_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
    print('Best epoch is {}'.format(best_epoch))
    
    

    plt.figure(figsize=(8, 5))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")

    plt.plot(acc_train_list, label = "train_acc.")
    plt.plot(acc_valid_list, label = "valid_acc.")
    plt.legend(loc=2)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(train_loss_list, label = "train_loss")
    plt.plot(valid_loss_list, label = "valid_loss")
    plt.legend(loc=2)
    plt.show()


# In[ ]:


#testing
#evaluation mode
model.eval()
result = []  #storing the result of prediction (probability of being positive/abnormal)
count = 0
with torch.no_grad(): #don't calculate the gradient
    for inputs in test_loader:
        # Move to device
        inputs = inputs.to(device)
        # Forward pass
        output = model.forward(inputs)
        count += 1
        print(count,'/',len(test_loader))
        #store the result
        result.append(np.exp(float((output[0][1]).cpu())))

#output the prediction
#generate list of image id
test_folder = 'data_path/test'
image_id = []
for i in range (len(test_paths)):
    image_id.append(test_paths[i][-len(test_paths[i])+len(test_folder)+1:-11])
#build data frame for pandas library
df = pd.DataFrame({'id':image_id,
                  'label':result})
#construct csv file for submission
#df.to_csv('/content/drive/My Drive/Colab Notebooks/submission.csv',index=False)

