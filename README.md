# P887Deep_learning_model_testing
## Overview 
This repository contains the development and testing of deep learning models like <br>
This repository contains code and resources for training and optimizing/tuning deep learning models like Convolution Neural Network (CNN), Long Short Term Memory (LSTM), Variational Autoencoder (VAE), and their hybrids, to predict and analyze sea surface temperature (SST) data. The dataset taken is daily netCDF data from 1982 to 2023 with a resolution of 0.25 for now. More tuning have to be done using 0.25 resolution data and later on 0.05 resolution data as well using different hyperparameters..<br>

**Why These Models<br>**

**1. Long Short-Term Memory (LSTM)<br>**
LSTMs excel in handling sequential data, making them ideal for modeling the temporal aspect of SST time series data.<br>
**How LSTM Works<br>**
LSTMs address the vanishing gradient problem faced by traditional Recurrent neural network (RNNs) by incorporating memory units with three gates:<br>

**Forget Gate:** Decides what information to discard from the cell state.<br>
**Input Gate:** Decides which values from the input will update the cell state.<br>
**Output Gate:** Controls the output and updates the hidden state.<br>
**Why LSTM?<br>**
LSTM networks are chosen for this task because they capture long-term dependencies in time series data, making them suitable for accurate forecasting over extended periods.<br>
For now, it is discussed about training, testing and preliminary results of LSTM.<br>

**2. Convolutional Neural Networks (CNN)<br>**
CNNs effectively capture spatial dependencies in the data, making them suitable for processing spatial data such as sea surface temperatures mapped over different geographic locations.<br>

**3. Variational Autoencoder (VAE)<br>**
VAEs can learn efficient data representations and generate synthetic data, which can be useful for anomaly detection and understanding variations in SST patterns.<br>

**4. Hybrid Models<br>**
Combining CNNs and LSTMs leverages both spatial and temporal features, providing a comprehensive approach to predicting SST.<br>

**Training, validation and testing dataset<br>**
Splitting of training, validation and testing data<br>
**Training Data:** 1982-2021 (80% training, 20% validation randomly selected)<br>
**Test Data:** 2022-2023<br>
**Dependencies<br>**
Python 3.9.7 (higher version to be upgraded)<br>
Keras API with TensorFlow backend<br>

**Model Training<br>**
Use the following hyperparameters for training our models:<br>

**Hyperparameters<br>**
**Learning Rate (LR): [0.001, 0.0001, 0.01]br**
Controls the speed of the learning process. Too high a value might overshoot the minima, while too low can lead to a long training process.<br>

**Epochs: [50, 100, 200, 400]<br>**
The number of complete passes through the training dataset. More epochs can lead to better training but might overfit if too many.<br>

**Batch Size: [32, 64, 128]<br>**
Number of samples per gradient update. Smaller batches provide a more accurate estimate of the gradient but are computationally expensive.<br>

**Number of Nodes in a Layer:<br>**
Determines the capacity of the model. More nodes can capture more complex patterns but also increase the risk of overfitting.<br>

**Number of Layers:5<br>**
More layers can capture deeper relationships in the data but also increase computational complexity and overfitting risk.<br>

**LSTM Units: [(64, 32, 16), (128, 64, 32), (256, 128, 64)]<br>**
The number of units in each LSTM layer. More units can model more complex relationships but at the cost of higher computational requirements.<br>

**Dropout: [0.1, 0.2]<br>**
Regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.<br>

**Sequence Length: [30, 60, 90]<br>**
The number of previous days used to predict the next day. Longer sequences provide more context but also require more memory and computational power.<br>

**Results<br>**
After training various models with the above hyperparameters, we observed the following optimal configuration as of now for our task (more testing to be done):<br>

Learning Rate: 0.0001<br>
Epochs: 200<br>
Batch Size: 64<br>
Number of Layers: 5<br>
LSTM Units: (256, 128, 64)]<br>
Dropout: 0.1<br>
Sequence Length: 60<br>
This configuration provided the best balance between training time and model accuracy, achieving the lowest validation error and highest test set performance (until now whatever testing is performed).<br>


