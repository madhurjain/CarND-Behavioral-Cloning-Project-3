## Behavioral Cloning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we use deep neural networks and convolutional neural networks to clone driving behavior. We train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

[//]: # (Image References)

[image0]: ./gifs/center_lane_driving.gif "Center Lane Driving"
[image1]: ./gifs/car_recovery.gif "Car Recovery"
[image2]: ./vimeo_link.png "Link to Vimeo"


[![Vimeo Link][image2]](https://vimeo.com/231326907)

---
### Summary

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

Project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I went ahead with the [nVidia Convolutional Neural Network Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) as the model proved to work perfectly. The model includes ReLU layers to introduce nonlinearity (code line 51), and the data is normalized in the model using a Keras lambda layer (code line 49).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting.

Initially, the data captured included the car driving only in the forward direction. Later, more data was generated by driving the car around in simulator in the reverse direction of the track and also capturing the data to bring the car on-track if it swerves outside the track. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 66).

#### 4. Appropriate training data

Training data was chosen to include left turns, right turns and recovery. I used a combination of center lane driving, both in forward and reverse direction of the track and also some data for recovering the car from the left and right sides of the road. The training data can be downloaded [here](https://s3-us-west-1.amazonaws.com/selfdriving/p3_drive_data.zip)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure the car drives in the center of the road at all times.

My first step was to use a convolution neural network model similar to the one used in Traffic Sign Classification project. By just changing the output to _1_ and loss function to _MSE_ I thought this model might be appropriate. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a very high mean squared error to begin with.

When the model was run in the simulator, the car drove towards the edge of the road and continued driving through the lake. So I completely replaced the model with the [nVidia CNN](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) as described in one of the videos. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 45-64) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					          | 
|:-----------------:|:-------------------------------------------:| 
| Input         		| 160x320x3 RGB image  						            | 
| Cropping 2D  	    | outputs 65x320x3                            |
| Normalize				  |	Lambda layer										            |
| Convolution 5x5   | 2x2 stride, ReLU activation                 |
| Convolution 5x5   | 2x2 stride, ReLU activation                 |
| Convolution 5x5   | 2x2 stride, ReLU activation                 |
| Convolution 3x3   | ReLU activation                             |
| Convolution 3x3   | ReLU activation                             |
| Flatten		        |                    			        						|
| Dropout  	        | Rate = 0.1                                  |
| Dense 100	        | 		                                        |
| Dense 50	        | 		                                        |
| Dropout  	        | Rate = 0.1                                  |
| Dense 10	        | 		                                        |
| Dense 1	          | 		                                        |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example of center lane driving:

![Center lane driving][image0]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover:

![Recovery of the car][image1]

Then I repeated this process while driving in opposite direction on the same track so I could get more data for right turns.

After the collection process, I had _7566_ data points. I then preprocessed this data by cropping the top pixels containing _trees and mountains_ and the bottom pixels containing the _car hood_ from the image. The image was also normalized using Keras Lambda function.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by increasing loss beyond that point. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Revision History

8/28/2017 - Added 2 dropout layers and increased the number of epochs to 8. This helped reduce the loss. Car drives wavy at the speed of 30.
8/27/2017 - Used the default nVidia CNN model without any dropout and trained for 3 epochs. Car drives smoothly at the speed of 20.