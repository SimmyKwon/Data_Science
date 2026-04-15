#%%Import packages
#Import the essential methods from PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
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

#Import Tenseal for the operation
import tenseal as ts

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

testdata_tmp = torchvision.datasets.FashionMNIST('./', train=False, transform=transformer, download=False)
# %%Extract 5000 images in total, 500 from each label to create Test dataset

indices = []
indices_val = []
for label in range(10):
    class_indices = torch.where(torch.tensor(testdata_tmp.targets) == label)[0]
    selected = torch.randperm(len(class_indices))[:500]
    indices.extend(class_indices[selected].tolist())

testdata = Subset(testdata_tmp, indices)

# %%Define the class labels for dataset
classes = list(testdata_tmp.classes)
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
        x = x*x
        # x.view() changes the shape of the given tensor
        x = x.view(-1, 5 * 14 * 14)
        x = self.forward1(x)
        x = x*x
        x = self.forward2(x)
        
        return x

#%%Declare model and criterions

#Make CNN instance
cnn = CNN()

logger.info(f'The structure of the neural network is as follows: \n {cnn}')
#Set the device onto CUDA if CUDA is available
device = torch.device('cpu')
logger.info(f'Device used for training and testing is: {device}')
#We will use BCEWithLogitsLoss as the problem is classification problem
criterion = nn.CrossEntropyLoss()
logger.info(f'Criterion for the model training is: \n {criterion}')
#Use Adam as the optimiser of learning rate of the cnn model
optimiser = optim.AdamW(cnn.parameters(), lr = 0.0005, weight_decay=0.001)
logger.info(f'Optimiser for the model training is: \n {optimiser}')
#Create a scheduler object so that at every 3rd step, the learning rate decreases for more detailed learning process
scheduler = optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=3, gamma = 0.1)
logger.info(f'The learning rate scheduler for the framework is: \n {scheduler.state_dict()}')
#Create the model parameter
CNNPARAM = 'Train_Test_Initial.pth'

#Load the model parameter 
cnn.load_state_dict(torch.load(f'./{CNNPARAM}'))
cnn.to(device)
cnn.eval()
#%%Create DataLoader for TVT

testloader = DataLoader(testdata, batch_size=1, shuffle=False)

# %%Make a EncConvModel that encapsulates cnn

class EncConvNet:
    def __init__(self, torch_nn):

        self.torch_nn = torch_nn

        self.conv1_weight = self.torch_nn.conv1.weight.data.view(
            self.torch_nn.conv1.out_channels, 
            self.torch_nn.conv1.kernel_size[0], self.torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = self.torch_nn.conv1.bias.data.tolist()
        
        self.forward1_weight = self.torch_nn.forward1.weight.T.data.tolist()
        self.forward1_bias = self.torch_nn.forward1.bias.data.tolist()
        
        self.forward2_weight = self.torch_nn.forward2.weight.T.data.tolist()
        self.forward2_bias = self.torch_nn.forward2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        #1st conv layer
        enc_channels_1 = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels_1.append(y)

        enc_x = ts.CKKSVector.pack_vectors(enc_channels_1)

        enc_x.square_()
        
        # forward1 layer
        enc_x = enc_x.mm(self.forward1_weight) + self.forward1_bias

        enc_x.square_()

        # forward2 layer
        enc_x = enc_x.mm(self.forward2_weight) + self.forward2_bias

        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

#%%Make the dictionary storing the result
predictions_dict = {GT : {preds : 0 for preds in classes} for GT in classes}

#%%Test function

def test(context, model, test_loader, kernel_shape, stride):

    logger.info('Test started!')
    test_start_time = time.time()

    #Disabling the gradients for evaluation
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data

            x_enc, windows_nb = ts.im2col_encoding(
            context, inputs.view(28, 28).tolist(), kernel_shape[0],
            kernel_shape[1], stride)

            outputs = model(x_enc, windows_nb)

            dec_outputs = outputs.decrypt()
            fin_output = torch.tensor(dec_outputs).view(1, -1)

            #Print the label with highest probability
            _, predictions = torch.max(fin_output, 1)

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
        try:
            precision = round(TP/(TP+FP),4)
            temp.append(precision)
        except:
            precision = 0
            temp.append(precision)
        logger.info(f'The precision for class "{label}" is: {100*precision}%')
        try:
            recall = round(TP /(TP+FN),4)
            temp.append(recall)
        except:
            recall = 0
            temp.append(recall)
        logger.info(f'The recall for class "{label}" is: {100*recall}%')
        try:
            f1_score = round((2*TP)/(2*TP+FP+FN),4)
            temp.append(f1_score)
        except:
            f1_score = 0
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

#%%
bits_scale = 26

context_token = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale,31]
)

# set the scale
context_token.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context_token.generate_galois_keys()

Enc_Model = EncConvNet(cnn)

#%%Run
test(context_token, Enc_Model, testloader, cnn.conv1.kernel_size, cnn.conv1.stride[0])

#%%