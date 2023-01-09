import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Define CNN Class
class convolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128,20)
        self.dropout = nn.Dropout(p=0.5)     

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(X.size(0), -1)
        X = self.dropout(F.relu(self.fc1(X)))
        X = self.fc2(X)
        return F.log_softmax(X,dim=1)
def main():
    root = '/Users/madankc/Desktop/marvel-images'
    transform = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])

    train_data = datasets.ImageFolder(os.path.join(root, 'train-data'), transform = transform)
    test_data = datasets.ImageFolder(os.path.join(root, 'test-data'), transform = transform)


    num_workers = 2
    valid_size = 0.15
    batch_size = 10
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split= int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_data, batch_size=batch_size,sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(train_data, batch_size=batch_size,sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size, num_workers=num_workers) 

    CNNModel = convolutionalNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CNNModel.parameters(), lr = 0.0001)

    n_epochs = 30
    train_losslist = []
    valid_loss_min = np.Inf # track change in validation loss

    import time
    start_time = time.time()
    for epoch in range(n_epochs):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        CNNModel.train()
        for data, target in train_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = CNNModel(data)

            # calculate the batch loss
            loss = criterion(output, target)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # update training loss
            train_loss += loss.item()*data.size(0)

        # validate the model         
        CNNModel.eval()
        for data, target in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = CNNModel(data)

            # calculate the batch loss
            loss = criterion(output, target)

            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        train_losslist.append(train_loss)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))


        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(CNNModel.state_dict(), 'model_marvel_heroes.pt')
            valid_loss_min = valid_loss

    #Testing the accuracy on test images
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = CNNModel(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    total_time = time.time() - start_time
    print(f'Total time taken: {total_time/60} minutes')
	    
if __name__ == '__main__':
    main()
