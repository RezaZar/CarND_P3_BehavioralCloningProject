# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Behavioral Cloning


---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./NVIDIA.jpg "NVIDIA's Model Structure"
[image2]: ./output_6_0.png "Sample Data 1"
[image3]: ./output_6_1.png "Sample Data 2"
[image4]: ./output_6_2.png "Sample Data 3"
[image5]: ./output_6_3.png "Sample Data 4"
[image6]: ./output_6_4.png "Sample Data 5"
[image7]: ./center.jpg "Sample Data 6"
[image8]: ./rec0.jpg "Sample Data 7"
[image9]: ./rec1.jpg "Sample Data 8"
[image10]: ./rec2.jpg "Sample Data 9"
[image11]: ./center2.jpg "Sample Data 10"
[image12]: ./center2_flipped.jpg "Sample Data 10"
[image13]: ./output_18_1.png "Sample Data 12"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* The jupyter notebook, CarND_P3_BehavioralCloning.ipynb, containing the code and results
* HTML version of the Jupyter notebook, CarND_P3_BehavioralCloning.html 
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The neural network used in this project is developed baes on the the NVIDIA end-to-end driving structure explained [here](https://arxiv.org/pdf/1604.07316v1.pdf) and illustrated below:

![alt text][image1]

The NVIDIA's architecture has of 5 convolutional layers followed by flattening, three fully connected layers and an output layer. In this project, the NVIDIA's architecture is modified in the following ways:

* A Lambda layer is added at the beginning to normalize the input images:

```python
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
```

* The input images are cropped using Cropping2D to remove 25 pixels from the bottom and 70 pixels from the top. By this cropping, only the most important area in the images are fed to the model:

```python
    model.add(Cropping2D(cropping=((70,25),(0,0))))
```

* A drop layer is added before the fully connected layers to achieve better generalization and avoid overfitting:

```python
    model.add(Dropout(0.5, noise_shape=None, seed=None))
```


#### 2. Attempts to reduce overfitting in the model

Reducing overfitting was considered while developing the model, training using and also after the model was trained:

* The data is split to train and validation data sets, and 20% of the data is used for validation. The train_test_split function from the sklearn.model_selection library is used to perform this:

```python
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
```

* The dropout layer in the model contributes to avoid overfitting:

```python
    model.add(Dropout(0.5, noise_shape=None, seed=None))
```

* While training the model, multiple runs were performed to achieve a reasonable number of epochs to train the model. First, 30 epochs were considered and validation loss was monitored. Although the training loss continued to reduce, the validation loss did not necessarily decrease consistently. Therefore, as explained later in this report, monitoring the validation loss and saving the best model was considered to achieve better generalization and avoid overfitting.

* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

* During the runs through the simulator, the autonomous driving was overridden manually and the control was then given back to the model to ensure that model could continue the driving and keep the vehicle between the lane lines.



#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Mean Squared Error is chosen as the loss function.
Model is trained for 20 epochs. Validation loss is monitored and the best model is saved. 
The snapshot of the model is provided below:

```python
    epochs = 20

    model.compile(loss='mse', optimizer='adam')

    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]

    history_object = model.fit_generator(train_generator, steps_per_epoch =
        (len(train_samples))/batch_size, validation_data = 
        validation_generator,
        validation_steps = len(validation_samples), 
        epochs=epochs, callbacks=callbacks_list, verbose=1)
```

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The default training data provided by Udacity was first used as a baseline. Since the simulations showed that the car had difficulties at specific curves and exited the road, further data collection was performed and was appended to the default data set.

During collection of further data, the following were considered:
* Driving on center of the lane
* Occassional driving closer to the inner curve at sharp curves
* Sharp steering at sharp curves to help with the recovery
* Using all three cameras and the flipped image from the center camera

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a model that was experimentally proven to be capable for self-driving missions. Therefore, the NVIDIA model was chosen and was fine tuned to meet the required performance for this project.

In order to gauge how well the model was working, The image and steering angle data were split into a training and validation set. The model was first trained for 30 epochs using the default data set. Mean squared error on the training set decreased throughout the 30 epochs but higher mean squared error on the validation did not decrease after initial epochs. This implied that the model was overfitting. 

To combat the overfitting, the model was modified by adding a dropout model and monitoring the validation accuracy. The trend of the validation accuracy improved after adding the dropout layer.

Additionally the model was saved only when the validation loss decreased. This allowed to used the model with the best validation accuracy for final simulation and testing.

The final step was to run the simulator to see how well the car was driving around track one. Although the model had a small train and validation loss, there were a few spots where the vehicle fell off the track. This was an indication that the default data was not sufficient. To improve the driving behavior in these cases, additional data was captured using the simulator to help the model recover as explained in the previous section.

Moreover, a corrective steering helped the model to recover from the edges. The correction is implemented in the data generator section and is applied to the steering angle images from the left and right cameras.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer            | Details                                                        | 
|:----------------:|:--------------------------------------------------------------:| 
| Input            | 320x160x3 images                                               | 
| Normalization    | Normalizing to (-1,1)                                          |
| Cropping         | 2D cropping, 70 pixels from the top, 20 pixels from the bottom |
| Convolution      | 5x5 convolution, 2x2 stride, 24 depth, Relu activation         |
| Convolution      | 5x5 convolution, 2x2 stride, 36 depth, Relu activation         |
| Convolution      | 5x5 convolution, 2x2 stride, 48 depth, Relu activation         |
| Convolution      | 5x5 convolution, 2x2 stride, 64 depth, Relu activation         |
| Convolution      | 5x5 convolution, 2x2 stride, 64 depth, Relu activation         |
| Flatten          | Flattened outputs                                              |
| Dropout          | 50% of the inputs dropped                                      |
| Dropout          | 100 outputs                                                    |
| Dropout          | 50  outputs                                                    |
| Dropout          | 10  outputs                                                    |
| Dropout          | 1 output, the steering angle                                   |

The architecture is similar to the NVIDIA's architecture explained earlier. Normalization, cropping and dropout layers are the main differences.

```python
    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())

    model.add(Dropout(0.5, noise_shape=None, seed=None))

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, in addition to the default dataset, an additional lap was recorded on track with more aggressive steering at sharp corners as well as center driving. 

Sample images from the augmented dataset are illustrated below:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

Here is an example of center driving:

![alt text][image7]

The additional captured data included recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like starting from the left lane marking:

![alt text][image8]

![alt text][image9]

![alt text][image10]

To augment the data sat, flipped images and angles from the center camera were appended to the dataset. This helps the model to avoid drifting to the left and generalize better. For example, here is an image that has then been flipped:

![alt text][image11]

![alt text][image12]

After the collection process, 12507 number of data point were available. The data was then pre-processed in the generator by adding a steering correction to the images from the right and left cameras to further help the recovery process. The generator also helped with managing the computational resource.  Additionally, the data is shuffled in the generator. Snapshopt of the generator code is provided below:

```python
    def generator(samples, batch_size=20):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                images = []
                angles = []

                for batch_sample in batch_samples:
                    source_path = batch_sample[0]
                    filename = source_path.split('/')[-1]  #center_2016_12_01_13_30_48_287.jpg
                    for index in ["center", "left", "right"]:
                        filename = filename.split('_')  #center_2016_12_01_13_30_48_287.jpg
                        filename[0]=index
                        filename='_'.join(filename)
                        image = cv2.imread(images_path + filename)
                        images.append(image)
                        if index == "center":
                            images.append(cv2.flip(image, 1))

                    correction = 0.3
                    angle = float(batch_sample[3])
                    angles.append(angle)
                    angles.append(-1.0 * angle)
                    angles.append(angle + correction)
                    angles.append(angle - correction)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)
```

As explained earlier, 20% of the data into a validation set. 

The training data was used to train the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 18 as evidenced by the graph below. An adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image13]

Finally, the best model in terms of validation loss was used in the generator to test the self-driving behavior. The video below shows that the model can successfully navigate the track, autonomously! 

[link to my video result](./run1.mp4)

