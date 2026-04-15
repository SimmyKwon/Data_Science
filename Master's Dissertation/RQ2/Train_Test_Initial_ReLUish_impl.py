#%%Import packages
#Import the essential methods from PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

#Import numpy for computation
import numpy as np
#Import os for jupyter notebook configuration and checking if the 
import os
#Import copy for copying an object
import copy
#Import random for controlling randomness
import random
#Import logging
import logging
#Import time for logging time
import time
#Import typing library for typing
from typing import Union

#Remove randomness of PyTorch CNN model by setting a seed value
torch.manual_seed(25)
random.seed(25)

#%%Set the directory to the file location
filename = os.path.dirname(__file__)
os.chdir(filename)
print(os.getcwd())

#And also name for logging
name = __file__.split('\\')[-1].split('.')[0]
print(name)
#%%Define transformers

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomHorizontalFlip(0.15),
    transforms.RandomVerticalFlip(0.15)
])


# %%
#Download training and test dataset from PyTorch repository with transformers applied
#The original image has size of 28*28 pixels
traindata = torchvision.datasets.FashionMNIST('./', train=True, transform=transformer, download=False)
testdata_tmp = torchvision.datasets.FashionMNIST('./', train=False, transform=transformer, download=False)
# %%Extract 5000 images in total, 500 from each label to create Test dataset

indices = []
indices_val = []
for label in range(10):
    class_indices = torch.where(torch.tensor(testdata_tmp.targets) == label)[0]
    selected = torch.randperm(len(class_indices))[:500]
    indices.extend(class_indices[selected].tolist())
    indices_val.extend(class_indices[~selected].tolist())

testdata = Subset(testdata_tmp, indices)
valdata = Subset(testdata_tmp, indices_val)

# %%Define the class labels for dataset
classes = list(traindata.classes)
print(classes)

#%%Create a logger object for recording the performance

logging.basicConfig(filename=f'{name}_logs.txt',
                    filemode='w',
                    format='Time: %(asctime)s,%(msecs)02d, Message: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger()

#Add stream handler for logging
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

#%%

ArrayLike = Union[torch.Tensor, np.ndarray]

def ReLU_approx(x = ArrayLike, coef = float) -> Union[torch.Tensor, np.ndarray]:
    
    if isinstance(x, np.ndarray):
        first_term = np.power(x, 3)
        second_term = np.power(x, 2)
        return (coef)*first_term + (1-coef)*second_term

    elif isinstance(x, torch.Tensor):
        first_term = torch.pow(x,3)
        second_term = torch.pow(x,2)
        return (coef)*first_term + (1-coef)*second_term

#%%Make model

#The parameters are adjusted accordingly to be properly trained and tested
class CNN(nn.Module):
    def __init__(self):
        #super() is defined so that the methods can be overriden from parent class to a child class
        super(CNN, self).__init__()
        #Conv2d is a 2-dimensional convolutional layer that extracts the features of each input layer from image
        #The image size becomes 28*28 from 14*14 due to stride, with 5 output channels from 1
        self.conv1 = nn.Conv2d(1,5, kernel_size=2, stride=2)
        #nn.Linear is a linear-connected neural network
        #From 980 nodes (14*14*5=980), there are now 140 nodes
        self.forward1 = nn.Linear(14*14*5,140)
        #From 140 nodes, there are now 10 nodes
        self.forward2 = nn.Linear(140,10)

    def forward(self,x):
        #The input first goes through convolutional layer, followed by ReLU activation function, and then maxpooled for 2 times
        x = self.conv1(x)
        x = ReLU_approx(x,0.6)
        # x.view() changes the shape of the given tensor
        x = x.view(-1, 5 * 14 * 14)
        x = self.forward1(x)
        x = ReLU_approx(x,0.6)
        x = self.forward2(x)
        
        return x

#%%Declare model and criterions

#Make CNN instance
cnn = CNN()
logger.info(f'The structure of the neural network is as follows: \n {cnn}')
fixpoint = "Fix me later!"
print(fixpoint)
logger.info(f'The activation function used in the network is: ReLU_approx')
#Set the device onto CUDA if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Device used for training and testing is: {device}')
#We will use BCEWithLogitsLoss as the problem is classification problem
criterion = nn.CrossEntropyLoss()
logger.info(f'Criterion for the model training is: \n {criterion}')
#Use Adam as the optimiser of learning rate of the cnn model
optimiser = optim.AdamW(cnn.parameters(), lr = 0.0005, weight_decay=0.01)
logger.info(f'Optimiser for the model training is: \n {optimiser}')
#Create a scheduler object so that at every 3rd step, the learning rate decreases for more detailed learning process
scheduler = optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=4, gamma = 0.1)
logger.info(f'The learning rate scheduler for the framework is: \n {scheduler.state_dict()}')
#Create the model parameter
CNNPARAM = f'{name}.pth'

#%%Create DataLoader for TVT

trainloader = DataLoader(traindata, batch_size=30, shuffle=True, pin_memory= True, persistent_workers=True, num_workers=6)
valloader = DataLoader(valdata, batch_size=10, shuffle=False, pin_memory= True, persistent_workers=True, num_workers=2)
testloader = DataLoader(testdata, batch_size=10, shuffle=False, pin_memory= True, persistent_workers=True, num_workers=2)

# %%

def train_and_val():
    if CNNPARAM in os.listdir('./'):
        pass
    else:
        #Send model 2~3 years NVIDIA GPU!
        cnn.to(device)
        #10 epochs for simplicity
        epoch = 25
        #Check loss for every nth batches
        check_batch = 250

        #Measure the starting time
        start_time = time.time()

        for k in range(epoch):
            cnn.train()
            logger.info(f'Epoch no.{k+1}')
            running_loss = 0
            for j, data in enumerate(trainloader):
                #Fetch the image tensors and labels from each data in dataloader
                inputs, targets = data
                #Send those inputs and labels onto cuda
                inputs, targets = inputs.cuda(), targets.cuda()
                optimiser.zero_grad()
                #Train the model with the image tensors, and get the outputs for loss computation
                outputs = cnn(inputs)

                #Compute the loss
                loss = criterion(outputs, targets)            
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                
                #Print the loss for every wanted batch
                if j % check_batch == (check_batch-1):
                    logger.info(f'Epoch number: {k + 1}, {j + 1}th batch loss: {running_loss / check_batch:.3f}')
                    running_loss = 0.0

            with torch.no_grad():
                cnn.train(False)
                val_loss_tot = 0

                for l, dat in enumerate(valloader):
                    inputs, targets = dat
                    #Send those inputs and labels onto cuda
                    inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = cnn(inputs)
                    val_loss = criterion(outputs, targets)
                    val_loss_tot += val_loss.item()

                logger.info(f'Epoch number: {k + 1}, validation loss: {val_loss_tot/len(valloader):.3f}')

            time.sleep(2)

        end_time = time.time()

        #Measure the total time in circuit
        total_time = end_time-start_time

        total_hr, rem = divmod(total_time,3600)
        total_min, total_sec = divmod(rem, 60)

        logger.info(f"Total time spent on training for {epoch} epoches: \n{total_hr:.0f} hours, {total_min:.0f} minutes, and {total_sec:.1f} seconds")
        #Save the model parameter in the local directory
        torch.save(cnn.state_dict(), f'./{CNNPARAM}')
        logger.info(f"The model parameter is saved in {os.getcwd()}")

#%%Make test function

def test():
    #Load the model parameter 
    cnn_test = CNN()
    cnn_test.load_state_dict(torch.load(f'./{CNNPARAM}'))
    cnn_test.to(device)

    #Make the dictionary storing the result
    predictions_dict = {GT : {preds : 0 for preds in classes} for GT in classes}

    #Turn the model into evaluation mode
    cnn_test.eval()

    logger.info('Test started!')
    test_start_time = time.time()

    #Disabling the gradients for evaluation
    with torch.no_grad():
        for data in testloader:
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = cnn_test(inputs)
            outputs = outputs.cuda()
            #Print the label with highest probability
            _, predictions = torch.max(outputs, 1)

            #For each GT and prediction, add the value onto the 
            for GT, prediction in zip(targets, predictions):
                predictions_dict[classes[GT]][classes[prediction]] += 1

    logger.info('Test finished!')
    test_end_time = time.time()

    total_test_time = test_end_time - test_start_time
    total_test_hr, rem2 = divmod(total_test_time,3600)
    total_test_min, total_test_sec = divmod(rem2, 60)

    logger.info(f"Total time spent on test: \n{total_test_hr:.0f} hours, {total_test_min:.0f} minutes, and {total_test_sec :.1f} seconds")

    #Measure accuracy, precision and recall for each class variable
    #Create a list containing metrics
    metrics = []

    overall_accuracy, overall_precision, overall_recall, overall_f1_score = 0,0,0,0

    for label in classes:
        #Make a temporary list that contains the accuracy, precision and recall value
        temp = []
        #Make a deepcopy of the existing classes list to perform list manipulation
        classes_new = copy.deepcopy(classes)
        classes_new.remove(label)

        #Compute True Positive - Number of times when the actual value and predicted value are the same
        TP = predictions_dict[label][label]

        #Compute False Positive and False Negative
        FP, FN = 0,0

        for sublabel in classes_new:
            FP += predictions_dict[sublabel][label]
            FN += predictions_dict[label][sublabel]
        
        #Compute False Negative
        TN = len(testdata) - (TP+FP+FN)

        #Compute accuracy, precision and recall, and append them onto temp
        accuracy = round((TP+TN)/(TP+TN+FP+FN),4)
        temp.append(accuracy)
        logger.info(f'The accuracy for class "{label}" is: {100*accuracy}%')
        precision = round(TP/(TP+FP),4)
        temp.append(precision)
        logger.info(f'The precision for class "{label}" is: {100*precision}%')
        recall = round(TP /(TP+FN),4)
        temp.append(recall)
        logger.info(f'The recall for class "{label}" is: {100*recall}%')
        f1_score = round((2*TP)/(2*TP+FP+FN),4)
        temp.append(f1_score)
        logger.info(f'The f1_score for class "{label}" is: {100*f1_score}%')
        metrics.append(temp)


    for elem in metrics:

        overall_accuracy += elem[0]
        overall_precision += elem[1]
        overall_recall += elem[2]
        overall_f1_score += elem[3]

    logger.info(f"The overall accuracy of the model is {100*overall_accuracy/len(classes):.2f}%")
    logger.info(f"The overall precision of the model is {100*overall_precision/len(classes):.2f}%")
    logger.info(f"The overall recall of the model is {100*overall_recall/len(classes):.2f}%")
    logger.info(f"The overall f1-score of the model is {100*overall_f1_score/len(classes):.2f}%")


#%%Run
if __name__ == '__main__':
    train_and_val()
    test()