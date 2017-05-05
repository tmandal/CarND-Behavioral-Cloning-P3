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

Each of 5 convolution layers and 3 fully connected layers use relu activation to add non-linearity to the model.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer at the output of penultimate fully connected layer in order to reduce overfitting. 

The entire dataset was split into training and validation samples with 20% set aside for validation. The model was trained with training samples only and the model was validated with validation samples that model did not see in the training. Achieving a low loss for validation samples ensures that model was not getting overfitted. The model was further tested by running it through the simulator in track 1 and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model was optimized on mean-square error using an adam optimizer, so the learning rate was not tuned manually. However, two other hyper-parameters, BATCHSIZE and NUMEPOCHS were tuned by doing multiple trainings with their different values. After testing the models trained with different parameters in simulator, BATCHSIZE=128 and NUMEPOCHS=5 were chosen as they gave optimal model. Lower BATCHSIZE gave suboptimal model which missed a couple of sharp turns in track 1. Training the model beyond 5 epochs did not improve the model performance much.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used driving data for center lane driving along with data from left and right cameras with some steering angle corrections. It turned out that steering angle correction of 0.10 for left and right camera images provided extra training data to train the model to run properly in simulator. Without these extra images with corrected steering angles, model was not able to learn to drive at the sharp turns of the roads. In addition, each image is flipped along with adjusted steering angle to augment the training data set. The flipped images provided a needed boost to learn to drive at the central region of the road.
















###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

