# dlp
#### based on the book `Deep Learning with Python`. Author: Francois Chollet. 
#### link for the book https://github.com/martianvenusian/books/blob/master/science_and_technology/programming/deep_learning/Deep%20Learning%20with%20Python.pdf

## dlp 5

#### Download kaggle cats and dogs dataset
##### https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip

##### The download cats and dogs datasets should be prepared as following:
```
kagglecatsanddogs/PetImages/cats/*.jpg
kagglecatsanddogs/PetImages/dogs/*.jpg
```
#### Prepare training, validation, test dataset
`$ python dlp_5.2_prepare_dataset.py`

### First Training
`$ python dlp_5.2.4_train.py`

```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               3211776   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________

Epoch 1/30
100/100 [==============================] - 20s 196ms/step - loss: 0.6920 - acc: 0.5400 - val_loss: 0.6837 - val_acc: 0.5750
Epoch 2/30
100/100 [==============================] - 20s 199ms/step - loss: 0.6684 - acc: 0.6115 - val_loss: 0.6568 - val_acc: 0.5890
Epoch 3/30
100/100 [==============================] - 18s 183ms/step - loss: 0.6298 - acc: 0.6605 - val_loss: 0.6720 - val_acc: 0.5900
Epoch 4/30
100/100 [==============================] - 18s 182ms/step - loss: 0.5823 - acc: 0.6930 - val_loss: 0.6018 - val_acc: 0.6740
Epoch 5/30
100/100 [==============================] - 18s 184ms/step - loss: 0.5476 - acc: 0.7260 - val_loss: 0.5862 - val_acc: 0.6890
Epoch 6/30
100/100 [==============================] - 18s 181ms/step - loss: 0.5144 - acc: 0.7430 - val_loss: 0.5794 - val_acc: 0.6980
Epoch 7/30
100/100 [==============================] - 18s 180ms/step - loss: 0.4832 - acc: 0.7645 - val_loss: 0.6057 - val_acc: 0.6780
Epoch 8/30
100/100 [==============================] - 18s 181ms/step - loss: 0.4581 - acc: 0.7890 - val_loss: 0.5702 - val_acc: 0.7060
Epoch 9/30
100/100 [==============================] - 19s 191ms/step - loss: 0.4269 - acc: 0.8110 - val_loss: 0.5629 - val_acc: 0.7190
Epoch 10/30
100/100 [==============================] - 19s 186ms/step - loss: 0.3935 - acc: 0.8290 - val_loss: 0.7176 - val_acc: 0.6560
Epoch 11/30
100/100 [==============================] - 19s 188ms/step - loss: 0.3851 - acc: 0.8165 - val_loss: 0.5499 - val_acc: 0.7330
Epoch 12/30
100/100 [==============================] - 19s 190ms/step - loss: 0.3592 - acc: 0.8375 - val_loss: 0.5681 - val_acc: 0.7150
Epoch 13/30
100/100 [==============================] - 19s 189ms/step - loss: 0.3248 - acc: 0.8625 - val_loss: 0.6324 - val_acc: 0.7100
Epoch 14/30
100/100 [==============================] - 18s 184ms/step - loss: 0.3080 - acc: 0.8705 - val_loss: 0.5754 - val_acc: 0.7310
Epoch 15/30
100/100 [==============================] - 19s 187ms/step - loss: 0.2770 - acc: 0.8765 - val_loss: 0.7333 - val_acc: 0.6960
Epoch 16/30
100/100 [==============================] - 21s 205ms/step - loss: 0.2558 - acc: 0.9030 - val_loss: 0.7129 - val_acc: 0.7070
Epoch 17/30
100/100 [==============================] - 20s 199ms/step - loss: 0.2282 - acc: 0.9030 - val_loss: 0.6511 - val_acc: 0.7340
Epoch 18/30
100/100 [==============================] - 20s 199ms/step - loss: 0.2056 - acc: 0.9205 - val_loss: 0.6227 - val_acc: 0.7240
Epoch 19/30
100/100 [==============================] - 20s 199ms/step - loss: 0.1803 - acc: 0.9350 - val_loss: 0.6541 - val_acc: 0.7350
Epoch 20/30
100/100 [==============================] - 20s 200ms/step - loss: 0.1618 - acc: 0.9375 - val_loss: 0.7334 - val_acc: 0.7200
Epoch 21/30
100/100 [==============================] - 20s 202ms/step - loss: 0.1442 - acc: 0.9500 - val_loss: 0.6884 - val_acc: 0.7330
Epoch 22/30
100/100 [==============================] - 20s 199ms/step - loss: 0.1191 - acc: 0.9620 - val_loss: 0.8748 - val_acc: 0.7140
Epoch 23/30
100/100 [==============================] - 20s 200ms/step - loss: 0.1095 - acc: 0.9655 - val_loss: 1.1969 - val_acc: 0.6790
Epoch 24/30
100/100 [==============================] - 20s 199ms/step - loss: 0.0961 - acc: 0.9720 - val_loss: 0.8720 - val_acc: 0.7200
Epoch 25/30
100/100 [==============================] - 20s 199ms/step - loss: 0.0848 - acc: 0.9710 - val_loss: 0.8594 - val_acc: 0.7240
Epoch 26/30
100/100 [==============================] - 21s 207ms/step - loss: 0.0703 - acc: 0.9820 - val_loss: 0.8550 - val_acc: 0.7320
Epoch 27/30
100/100 [==============================] - 20s 201ms/step - loss: 0.0606 - acc: 0.9835 - val_loss: 0.9462 - val_acc: 0.7400
Epoch 28/30
100/100 [==============================] - 20s 204ms/step - loss: 0.0501 - acc: 0.9880 - val_loss: 0.9250 - val_acc: 0.7350
Epoch 29/30
100/100 [==============================] - 20s 203ms/step - loss: 0.0387 - acc: 0.9890 - val_loss: 1.0429 - val_acc: 0.7220
Epoch 30/30
100/100 [==============================] - 20s 201ms/step - loss: 0.0367 - acc: 0.9885 - val_loss: 1.1217 - val_acc: 0.7100

```
### Result
overfitting

![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.4_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.4_2.png?raw=true)


### Data augmentation
`$ python dlp_5.2.5_data_augmentation.py`

An example for data augmentation
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.5_data_augmentation_0.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.5_data_augmentation_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.5_data_augmentation_2.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.5_data_augmentation_3.png?raw=true)

### Train with data augmentation and dropout
`$ python dlp_5.2.5_train_with_data_aug_dropout.py`

```
Epoch 1/30
100/100 [==============================] - 18s 176ms/step - loss: 0.6934 - acc: 0.5105 - val_loss: 0.6777 - val_acc: 0.5540
Epoch 2/30
100/100 [==============================] - 18s 178ms/step - loss: 0.6786 - acc: 0.5700 - val_loss: 0.6679 - val_acc: 0.5680
Epoch 3/30
100/100 [==============================] - 18s 181ms/step - loss: 0.6701 - acc: 0.5805 - val_loss: 0.6469 - val_acc: 0.6030
Epoch 4/30
100/100 [==============================] - 18s 180ms/step - loss: 0.6573 - acc: 0.6040 - val_loss: 0.6373 - val_acc: 0.6280
Epoch 5/30
100/100 [==============================] - 18s 181ms/step - loss: 0.6341 - acc: 0.6385 - val_loss: 0.6435 - val_acc: 0.6090
Epoch 6/30
100/100 [==============================] - 20s 199ms/step - loss: 0.6265 - acc: 0.6440 - val_loss: 0.6118 - val_acc: 0.6580
Epoch 7/30
100/100 [==============================] - 20s 198ms/step - loss: 0.6061 - acc: 0.6760 - val_loss: 0.5813 - val_acc: 0.6900
Epoch 8/30
100/100 [==============================] - 20s 197ms/step - loss: 0.6125 - acc: 0.6505 - val_loss: 0.5767 - val_acc: 0.6970
Epoch 9/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5976 - acc: 0.6740 - val_loss: 0.6147 - val_acc: 0.6570
Epoch 10/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5937 - acc: 0.6910 - val_loss: 0.5655 - val_acc: 0.7070
Epoch 11/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5902 - acc: 0.6995 - val_loss: 0.5978 - val_acc: 0.6550
Epoch 12/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5804 - acc: 0.7010 - val_loss: 0.5583 - val_acc: 0.7030
Epoch 13/30
100/100 [==============================] - 20s 201ms/step - loss: 0.5671 - acc: 0.7075 - val_loss: 0.6016 - val_acc: 0.6620
Epoch 14/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5619 - acc: 0.7180 - val_loss: 0.5493 - val_acc: 0.7130
Epoch 15/30
100/100 [==============================] - 20s 197ms/step - loss: 0.5673 - acc: 0.7135 - val_loss: 0.5706 - val_acc: 0.6990
Epoch 16/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5565 - acc: 0.7200 - val_loss: 0.5349 - val_acc: 0.7220
Epoch 17/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5570 - acc: 0.7180 - val_loss: 0.5123 - val_acc: 0.7320
Epoch 18/30
100/100 [==============================] - 20s 200ms/step - loss: 0.5575 - acc: 0.7175 - val_loss: 0.5322 - val_acc: 0.7220
Epoch 19/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5461 - acc: 0.7260 - val_loss: 0.5169 - val_acc: 0.7310
Epoch 20/30
100/100 [==============================] - 20s 199ms/step - loss: 0.5459 - acc: 0.7335 - val_loss: 0.5329 - val_acc: 0.7370
Epoch 21/30
100/100 [==============================] - 20s 200ms/step - loss: 0.5415 - acc: 0.7235 - val_loss: 0.5253 - val_acc: 0.7380
Epoch 22/30
100/100 [==============================] - 20s 200ms/step - loss: 0.5277 - acc: 0.7320 - val_loss: 0.5315 - val_acc: 0.7310
Epoch 23/30
100/100 [==============================] - 21s 209ms/step - loss: 0.5279 - acc: 0.7330 - val_loss: 0.5412 - val_acc: 0.7290
Epoch 24/30
100/100 [==============================] - 20s 203ms/step - loss: 0.5314 - acc: 0.7320 - val_loss: 0.4911 - val_acc: 0.7530
Epoch 25/30
100/100 [==============================] - 21s 208ms/step - loss: 0.5408 - acc: 0.7330 - val_loss: 0.5414 - val_acc: 0.7150
Epoch 26/30
100/100 [==============================] - 21s 207ms/step - loss: 0.5329 - acc: 0.7340 - val_loss: 0.5349 - val_acc: 0.7340
Epoch 27/30
100/100 [==============================] - 21s 212ms/step - loss: 0.5167 - acc: 0.7460 - val_loss: 0.4882 - val_acc: 0.7660
Epoch 28/30
100/100 [==============================] - 20s 205ms/step - loss: 0.5130 - acc: 0.7490 - val_loss: 0.5164 - val_acc: 0.7360
Epoch 29/30
100/100 [==============================] - 21s 210ms/step - loss: 0.5179 - acc: 0.7360 - val_loss: 0.5035 - val_acc: 0.7450
Epoch 30/30
100/100 [==============================] - 21s 206ms/step - loss: 0.4948 - acc: 0.7630 - val_loss: 0.4823 - val_acc: 0.7590
```
### Output
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.4_data_augment_dropout_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.4_data_augment_dropout_2.png?raw=true)