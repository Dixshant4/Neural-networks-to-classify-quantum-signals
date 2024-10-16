# Neural-networks-to-classify-quantum-signals
32 independently trained models to classify a 32-bit string from a pool of 2^32 = 4,294,967,296 different classes. 


For the scope of this research to have any practical application, it is necessary to be able to accurately classify from a large pool of classes. Imagine a 3-bit string of 0s and 1s. (For example 000 or 010). The number of possible combinations of this 3-bit string is 23=8. If we use the intensity in optical signals to encode this 3-bit string, is it possible to know which of the 8 possible strings one receives? This is the question that will be answered.

Instead of training one large complex model to predict 8 different classes, 3 independent models were trained to predict either 0 or 1 for each index of the string. I.e. Model 1 will predict whether the zeroth index of the string ‘000’ is a 0 or a 1. 

Theoretically, such a method could be used to show that a 32-bit string can be accurately classified from a pool of 2^32 = 4,294,967,296 different strings using just 32 independently trained models. 

In general, if we want to solve this using machine learning, one approach is to train a classifier that predicts 4,294,967,296  classes. For this, we would need huge amounts of data (on the order of 1M) and the probability of predicting a certain class will be more spread out. This will make the model less accurate especially when the number of classes to predict increases.

An alternative approach involves breaking this problem down to multiple binary classification tasks and training multiple neural networks simultaneously. By doing so, a 95% accuracy was achieved for this task.
