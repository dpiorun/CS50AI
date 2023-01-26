# Background
In the project the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) dataset is used, which contains thousands of images of 43 different kinds of road signs.  
Download the [data set](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip) for this project and unzip it. Move the resulting gtsrb directory inside of your traffic directory.

# Experimentation process
## First model
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 6272)              0         
                                                                 
 dense (Dense)               (None, 128)               802944    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 43)                5547      
                                                                 
=================================================================
```
The first model wasn't successful at all. A single convolution layer seems not to be sufficient. The accuracy of this model achieved 5,5%. What is worth noticing, after increasing the number of filters in the convolutional layer from 32 to 200 the accuracy raised to 28%.

## Second model
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 12, 12, 32)        9248      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 6, 32)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 128)               147584    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 43)                5547      
                                                                 
=================================================================
```
This model trained in 10 epochs reaches an accuracy of 89.81%. This is definitely too less to be worth noticing.

## Third model
```
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        896       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 12, 12, 32)        9248      
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 6, 6, 32)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 4, 4, 32)          9248      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 2, 2, 32)         0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 128)               0         
                                                                 
 dense (Dense)               (None, 128)               16512     
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 43)                5547      
                                                                 
=================================================================
```
Adding another convolutional layer appears to decrease the accuracy to 86.79%.

## Fourth model
```
Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 200)       5600      
                                                                 
 conv2d_1 (Conv2D)           (None, 26, 26, 200)       360200    
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 200)      0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 33800)             0         
                                                                 
 dense (Dense)               (None, 128)               4326528   
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 43)                5547      
                                                                 
```
In this model, I wanted to see how pooling infers accuracy. This model achieves an accuracy of 95%. Training the model in more epochs (I tried 20 and 30) doesn't bring spectacular improvement. With 30 epochs the model reaches an accuracy of 97%. Increasing the number of filters in the second convolutional layer (from 200 to 350) and in the hidden fully connected layer (from 128 to 400) actually decreased the accuracy to 93%.


## Final model
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 46, 46, 200)       5600      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 23, 23, 200)      0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 21, 21, 250)       450250    
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 10, 10, 250)      0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 8, 8, 350)         787850    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 350)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 5600)              0         
                                                                 
 dense (Dense)               (None, 400)               2240400   
                                                                 
 dropout (Dropout)           (None, 400)               0         
                                                                 
 dense_1 (Dense)             (None, 43)                17243     
                                                                 
=================================================================
```
After experimenting with a few settings on this model, which gave an accuracy slightly below 99% I decided to increase the size of the input image to 48x48 and create three convolutional layers with an increasing number of filters. The best accuracy is achieved for a kernel 3x3 in each convolutional layer. This model in the 10th epoch reaches an accuracy of 98.76%, but in the 28th it's already 99.62%. The evaluation of the model ran on the entire dataset achieves the result: 
```
833/833 - 158s - loss: 0.0420 - accuracy: 0.9962 - 158s/epoch - 190ms/step
```
Which I believe is a good point to start.
In this repository `model.h5` is the trained model, which achieves an accuracy of 99.62 % on a GTSRB dataset.

# Evaluation.py
To avaluate a saved model on the dataset run:
```
python3 evaluate.py model.h5 data_directory
```