## MSc Dissertation

This directory contains the code that I created for my master's thesis.

My master's was done at university of Royal Holloway, and I explored the methods of optimising Fully-Homomorphic Encryption on Deep Learning Frameworks.

There are 3 main RQs in my dissertation, which are:

**RQ1**: Investigate the different parameters of the CKKS scheme, and apply various combinations of these parameters on encrypting the image dataset that is used to test deep learning frameworks to discover if the parameter change affects the model’s inference, and find out the parameters that affect the model's performance significantly.

**RQ2**: To discover the alternatives of $f(x) = x^2$, which is a function commonly used as an activation function in neural networks that are created to be tested on CKKS encrypted dataset, to apply the widely used activation functions, such as ReLU, approximately. Activation function computes each neuron’s output values (activation) based on inputs in neural networks, and therefore decides the model’s ability to learn complex patterns. It is important to set the right activation for the best performance of the neural network built.

**RQ3**: To suggest alternative ways of inferring encrypted data to reduce the time of inference while making the test data resilient to attacks. 

The dataset used for the investigation was FashioMNIST dataset for simplicity and compatibility with the FHE scheme used, and the scheme used in the experiments was CKKS scheme. I used Tenseal library in Python (https://github.com/OpenMined/TenSEAL) to define and apply the FHE scheme on my dataset.

Each folder in the directory corresponds to the aforementioned RQs.

Full dissertation available upon request.
