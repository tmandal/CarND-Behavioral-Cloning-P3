# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the data supplied in the project to emulate good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

My project is accessible from https://github.com/tmandal/CarND-Behavioral-Cloning-P3

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model closely follows NVIDIA's end-to-end model for self driving cars. 
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

The model uses the following layers in series.
* Normalization layer to restrict image feature values to be in [-1.0, 1.0] range.
* Crop layer to remove top 50 and bottom 20 pixel rows to get rid of information unrelated to road view.
* 3 5x5 strided convolution layers with 2x2 strides for 12, 24, 48 filters in each layer respectively.
* 2 3x3 convolution layers for 64 filters in each layer.
* 3 fully connected layers with 100, 50, 10 neurons in each layer respectively.
* Dropout layer with 50% dropout probability 
* Final fully connected layer to output steering angle

Each of 5 convolution layers and 3 fully connected layers uses relu activation to add non-linearity to the model.

The configurations for each network layer are described in below table.

Layer (type)                 |Output Shape              |Param #   
--- | --- | ---
lambda_1 (Lambda)            |(None, 160, 320, 3)       |0         
cropping2d_1 (Cropping2D)    |(None, 90, 320, 3)        |0         
conv2d_1 (Conv2D)            |(None, 43, 158, 24)       |1824      
activation_1 (Activation)    |(None, 43, 158, 24)       |0         
conv2d_2 (Conv2D)            |(None, 20, 77, 36)        |21636     
activation_2 (Activation)    |(None, 20, 77, 36)        |0         
conv2d_3 (Conv2D)            |(None, 8, 37, 48)         |43248   
activation_3 (Activation)    |(None, 8, 37, 48)         |0         
conv2d_4 (Conv2D)            |(None, 6, 35, 64)         |27712     
activation_4 (Activation)    |(None, 6, 35, 64)         |0         
conv2d_5 (Conv2D)            |(None, 4, 33, 64)         |36928     
activation_5 (Activation)    |(None, 4, 33, 64)         |0         
flatten_1 (Flatten)          |(None, 8448)              |0         
dense_1 (Dense)              |(None, 100)               |844900    
activation_6 (Activation)    |(None, 100)               |0         
dense_2 (Dense)              |(None, 50)                |5050      
activation_7 (Activation)    |(None, 50)                |0         
dense_3 (Dense)              |(None, 10)                |510       
dropout_1 (Dropout)          |(None, 10)                |0         
activation_8 (Activation)    |(None, 10)                |0         
dense_4 (Dense)              |(None, 1)                 |11        

Total number of trainable parameters in this network is approximately one million. 

I started with NVIDIA's end-to-end model with the addition of a couple of pooling layers to reduce overall parameter count. This modified model was able to learn from training dataset provided in this problem. But the model could not steer the car safely at sharp turns. As I augmented the training dataset with left and right camera images, the model was able to learn few of sharp turns but it eventually failed at rest of the sharp turns. Eventually I had to remove pooling layers to have the model learn more accurate features in order to learn how to drive around all curves.

The model was trained on a GPU and each epoch took two minutes to complete its training.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer at the output of penultimate fully connected layer in order to reduce overfitting. 

The entire dataset was split into training and validation samples with 20% set aside for validation. The model was trained with training samples only and the model was validated with validation samples that model did not see in the training. Achieving a low loss for validation samples ensures that model was not getting overfitted. The model was further tested by running it through the simulator in track 1 and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model was optimized on mean-square error using an adam optimizer, so the learning rate was not tuned manually. However, two other hyper-parameters, BATCHSIZE and NUMEPOCHS were tuned by doing multiple trainings with their different values. After testing the models trained with different parameters in simulator, BATCHSIZE=64 or 128 and NUMEPOCHS=5 were chosen as they gave optimal model. Lower BATCHSIZE gave suboptimal model which missed a couple of sharp turns in track 1. Training the model beyond 5 epochs did not improve the model performance much.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used driving data for center lane driving along with data from left and right cameras with some steering angle corrections. It turned out that steering angle correction of 0.10 for left and right camera images provided extra training data to train the model to run properly in simulator. Without these extra images with corrected steering angles, model was not able to learn to drive at the sharp turns of the roads. In addition, each image is flipped along with adjusted steering angle to augment the training data set. The flipped images provided a needed boost to learn to drive at the central region of the road.

[image_histo]: examples/histo3.jpg
[image_sample_center_noflip]: examples/sample_center_flipFalse.jpg
[image_sample_center_noflip_cropped]: examples/sample_center_flipFalse_cropped.jpg
[image_sample_left_noflip]: examples/sample_left_flipFalse.jpg
[image_sample_right_noflip]: examples/sample_right_flipFalse.jpg
[image_sample_center_flip]: examples/sample_center_flipTrue.jpg

![alt text][image_sample_center_noflip]
![alt text][image_sample_center_noflip_cropped]


*Histogram of udacity provided data after augmentation*
![alt text][image_histo]
#### 5. Conclusions

The supplied dataset along with augmented data from left and right cameras with flips helped train a NVIDIA's drivenet based model to learn to drive on track 1. The trained model was able to successfully drive the car in track 1 at 25 mph in the supplied simulator without any problems. The model was tried to complete many laps in track 1 for 30 minutes and there was no unsafe driving observed with this model.
