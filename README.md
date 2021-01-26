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
#### output
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
Epoch 1/100
100/100 [==============================] - 19s 192ms/step - loss: 0.6936 - acc: 0.5080 - val_loss: 0.6818 - val_acc: 0.5770
Epoch 2/100
100/100 [==============================] - 20s 198ms/step - loss: 0.6811 - acc: 0.5600 - val_loss: 0.6605 - val_acc: 0.5740
Epoch 3/100
100/100 [==============================] - 20s 198ms/step - loss: 0.6683 - acc: 0.5880 - val_loss: 0.6528 - val_acc: 0.6020
Epoch 4/100
100/100 [==============================] - 20s 198ms/step - loss: 0.6587 - acc: 0.5995 - val_loss: 0.6369 - val_acc: 0.6290
Epoch 5/100
100/100 [==============================] - 20s 202ms/step - loss: 0.6378 - acc: 0.6330 - val_loss: 0.6192 - val_acc: 0.6560
Epoch 6/100
100/100 [==============================] - 20s 201ms/step - loss: 0.6241 - acc: 0.6380 - val_loss: 0.5948 - val_acc: 0.6700
Epoch 7/100
100/100 [==============================] - 20s 203ms/step - loss: 0.6063 - acc: 0.6675 - val_loss: 0.5919 - val_acc: 0.6730
Epoch 8/100
100/100 [==============================] - 20s 199ms/step - loss: 0.6077 - acc: 0.6605 - val_loss: 0.5858 - val_acc: 0.6740
Epoch 9/100
100/100 [==============================] - 20s 199ms/step - loss: 0.6074 - acc: 0.6755 - val_loss: 0.6223 - val_acc: 0.6630
Epoch 10/100
100/100 [==============================] - 20s 199ms/step - loss: 0.5850 - acc: 0.6765 - val_loss: 0.5616 - val_acc: 0.7030
Epoch 11/100
100/100 [==============================] - 20s 200ms/step - loss: 0.5914 - acc: 0.6850 - val_loss: 0.5988 - val_acc: 0.6710
Epoch 12/100
100/100 [==============================] - 18s 179ms/step - loss: 0.5829 - acc: 0.6980 - val_loss: 0.5634 - val_acc: 0.7000
Epoch 13/100
100/100 [==============================] - 18s 180ms/step - loss: 0.5762 - acc: 0.7005 - val_loss: 0.5610 - val_acc: 0.7080
Epoch 14/100
100/100 [==============================] - 18s 181ms/step - loss: 0.5711 - acc: 0.6955 - val_loss: 0.5353 - val_acc: 0.7250
Epoch 15/100
100/100 [==============================] - 18s 178ms/step - loss: 0.5592 - acc: 0.7070 - val_loss: 0.5469 - val_acc: 0.7200
Epoch 16/100
100/100 [==============================] - 18s 178ms/step - loss: 0.5562 - acc: 0.7095 - val_loss: 0.5443 - val_acc: 0.7250
Epoch 17/100
100/100 [==============================] - 18s 178ms/step - loss: 0.5475 - acc: 0.7195 - val_loss: 0.5411 - val_acc: 0.7280
Epoch 18/100
100/100 [==============================] - 18s 181ms/step - loss: 0.5437 - acc: 0.7235 - val_loss: 0.5382 - val_acc: 0.7190
Epoch 19/100
100/100 [==============================] - 18s 179ms/step - loss: 0.5483 - acc: 0.7090 - val_loss: 0.5694 - val_acc: 0.6860
Epoch 20/100
100/100 [==============================] - 18s 181ms/step - loss: 0.5400 - acc: 0.7220 - val_loss: 0.5264 - val_acc: 0.7380
Epoch 21/100
100/100 [==============================] - 18s 180ms/step - loss: 0.5286 - acc: 0.7440 - val_loss: 0.5358 - val_acc: 0.7180
Epoch 22/100
100/100 [==============================] - 18s 181ms/step - loss: 0.5489 - acc: 0.7190 - val_loss: 0.5441 - val_acc: 0.7160
Epoch 23/100
100/100 [==============================] - 18s 180ms/step - loss: 0.5349 - acc: 0.7330 - val_loss: 0.5073 - val_acc: 0.7520
Epoch 24/100
100/100 [==============================] - 18s 183ms/step - loss: 0.5278 - acc: 0.7300 - val_loss: 0.5049 - val_acc: 0.7560
Epoch 25/100
100/100 [==============================] - 20s 197ms/step - loss: 0.5333 - acc: 0.7340 - val_loss: 0.5160 - val_acc: 0.7440
Epoch 26/100
100/100 [==============================] - 20s 196ms/step - loss: 0.5117 - acc: 0.7525 - val_loss: 0.5181 - val_acc: 0.7480
Epoch 27/100
100/100 [==============================] - 20s 195ms/step - loss: 0.5269 - acc: 0.7340 - val_loss: 0.5441 - val_acc: 0.7160
Epoch 28/100
100/100 [==============================] - 20s 196ms/step - loss: 0.5212 - acc: 0.7470 - val_loss: 0.5505 - val_acc: 0.7140
Epoch 29/100
100/100 [==============================] - 20s 196ms/step - loss: 0.5129 - acc: 0.7425 - val_loss: 0.5042 - val_acc: 0.7470
Epoch 30/100
100/100 [==============================] - 20s 200ms/step - loss: 0.5227 - acc: 0.7390 - val_loss: 0.5465 - val_acc: 0.7160
Epoch 31/100
100/100 [==============================] - 20s 199ms/step - loss: 0.5091 - acc: 0.7335 - val_loss: 0.5089 - val_acc: 0.7370
Epoch 32/100
100/100 [==============================] - 20s 204ms/step - loss: 0.5128 - acc: 0.7535 - val_loss: 0.4956 - val_acc: 0.7520
Epoch 33/100
100/100 [==============================] - 21s 209ms/step - loss: 0.5072 - acc: 0.7430 - val_loss: 0.4839 - val_acc: 0.7630
Epoch 34/100
100/100 [==============================] - 20s 199ms/step - loss: 0.5041 - acc: 0.7455 - val_loss: 0.4810 - val_acc: 0.7720
Epoch 35/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4875 - acc: 0.7635 - val_loss: 0.5244 - val_acc: 0.7490
Epoch 36/100
100/100 [==============================] - 20s 198ms/step - loss: 0.4963 - acc: 0.7580 - val_loss: 0.5139 - val_acc: 0.7360
Epoch 37/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4968 - acc: 0.7645 - val_loss: 0.4829 - val_acc: 0.7650
Epoch 38/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4866 - acc: 0.7655 - val_loss: 0.4733 - val_acc: 0.7900
Epoch 39/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4900 - acc: 0.7690 - val_loss: 0.4715 - val_acc: 0.7830
Epoch 40/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4800 - acc: 0.7620 - val_loss: 0.4890 - val_acc: 0.7700
Epoch 41/100
100/100 [==============================] - 19s 195ms/step - loss: 0.4878 - acc: 0.7685 - val_loss: 0.4670 - val_acc: 0.7870
Epoch 42/100
100/100 [==============================] - 20s 196ms/step - loss: 0.4937 - acc: 0.7615 - val_loss: 0.4879 - val_acc: 0.7650
Epoch 43/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4809 - acc: 0.7655 - val_loss: 0.4842 - val_acc: 0.7760
Epoch 44/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4736 - acc: 0.7765 - val_loss: 0.4731 - val_acc: 0.7770
Epoch 45/100
100/100 [==============================] - 20s 198ms/step - loss: 0.4617 - acc: 0.7850 - val_loss: 0.4894 - val_acc: 0.7690
Epoch 46/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4680 - acc: 0.7790 - val_loss: 0.4682 - val_acc: 0.7770
Epoch 47/100
100/100 [==============================] - 20s 198ms/step - loss: 0.4696 - acc: 0.7735 - val_loss: 0.4911 - val_acc: 0.7780
Epoch 48/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4613 - acc: 0.7790 - val_loss: 0.4720 - val_acc: 0.7810
Epoch 49/100
100/100 [==============================] - 20s 200ms/step - loss: 0.4629 - acc: 0.7750 - val_loss: 0.4896 - val_acc: 0.7770
Epoch 50/100
100/100 [==============================] - 20s 203ms/step - loss: 0.4700 - acc: 0.7795 - val_loss: 0.4780 - val_acc: 0.7770
Epoch 51/100
100/100 [==============================] - 20s 203ms/step - loss: 0.4677 - acc: 0.7825 - val_loss: 0.4649 - val_acc: 0.7890
Epoch 52/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4483 - acc: 0.7880 - val_loss: 0.5939 - val_acc: 0.7120
Epoch 53/100
100/100 [==============================] - 18s 182ms/step - loss: 0.4572 - acc: 0.7690 - val_loss: 0.4596 - val_acc: 0.7880
Epoch 54/100
100/100 [==============================] - 18s 184ms/step - loss: 0.4435 - acc: 0.7880 - val_loss: 0.4633 - val_acc: 0.7890
Epoch 55/100
100/100 [==============================] - 18s 181ms/step - loss: 0.4448 - acc: 0.7945 - val_loss: 0.5179 - val_acc: 0.7450
Epoch 56/100
100/100 [==============================] - 18s 181ms/step - loss: 0.4483 - acc: 0.7990 - val_loss: 0.4566 - val_acc: 0.7930
Epoch 57/100
100/100 [==============================] - 18s 181ms/step - loss: 0.4386 - acc: 0.7960 - val_loss: 0.4975 - val_acc: 0.7750
Epoch 58/100
100/100 [==============================] - 18s 181ms/step - loss: 0.4574 - acc: 0.7830 - val_loss: 0.4739 - val_acc: 0.7960
Epoch 59/100
100/100 [==============================] - 18s 178ms/step - loss: 0.4420 - acc: 0.7950 - val_loss: 0.5701 - val_acc: 0.7430
Epoch 60/100
100/100 [==============================] - 18s 178ms/step - loss: 0.4342 - acc: 0.7985 - val_loss: 0.4552 - val_acc: 0.7970
Epoch 61/100
100/100 [==============================] - 18s 180ms/step - loss: 0.4274 - acc: 0.8065 - val_loss: 0.4581 - val_acc: 0.7950
Epoch 62/100
100/100 [==============================] - 18s 179ms/step - loss: 0.4411 - acc: 0.7930 - val_loss: 0.4918 - val_acc: 0.7740
Epoch 63/100
100/100 [==============================] - 18s 180ms/step - loss: 0.4264 - acc: 0.7975 - val_loss: 0.4621 - val_acc: 0.8070
Epoch 64/100
100/100 [==============================] - 18s 179ms/step - loss: 0.4243 - acc: 0.8130 - val_loss: 0.4464 - val_acc: 0.7970
Epoch 65/100
100/100 [==============================] - 19s 186ms/step - loss: 0.4299 - acc: 0.7990 - val_loss: 0.5039 - val_acc: 0.7600
Epoch 66/100
100/100 [==============================] - 21s 208ms/step - loss: 0.4241 - acc: 0.7960 - val_loss: 0.4494 - val_acc: 0.8020
Epoch 67/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4264 - acc: 0.7945 - val_loss: 0.4398 - val_acc: 0.7910
Epoch 68/100
100/100 [==============================] - 20s 201ms/step - loss: 0.4233 - acc: 0.8085 - val_loss: 0.4538 - val_acc: 0.7970
Epoch 69/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4298 - acc: 0.7890 - val_loss: 0.4457 - val_acc: 0.8030
Epoch 70/100
100/100 [==============================] - 20s 198ms/step - loss: 0.4204 - acc: 0.7955 - val_loss: 0.4591 - val_acc: 0.7970
Epoch 71/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4194 - acc: 0.8085 - val_loss: 0.4615 - val_acc: 0.7890
Epoch 72/100
100/100 [==============================] - 20s 198ms/step - loss: 0.4162 - acc: 0.8080 - val_loss: 0.4608 - val_acc: 0.7970
Epoch 73/100
100/100 [==============================] - 20s 198ms/step - loss: 0.4113 - acc: 0.8130 - val_loss: 0.4709 - val_acc: 0.7860
Epoch 74/100
100/100 [==============================] - 19s 195ms/step - loss: 0.4100 - acc: 0.8050 - val_loss: 0.4472 - val_acc: 0.8090
Epoch 75/100
100/100 [==============================] - 20s 195ms/step - loss: 0.4153 - acc: 0.8050 - val_loss: 0.4290 - val_acc: 0.8050
Epoch 76/100
100/100 [==============================] - 20s 195ms/step - loss: 0.4124 - acc: 0.8160 - val_loss: 0.4531 - val_acc: 0.8120
Epoch 77/100
100/100 [==============================] - 20s 196ms/step - loss: 0.4119 - acc: 0.8245 - val_loss: 0.4723 - val_acc: 0.7940
Epoch 78/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4003 - acc: 0.8125 - val_loss: 0.4857 - val_acc: 0.8040
Epoch 79/100
100/100 [==============================] - 20s 196ms/step - loss: 0.3976 - acc: 0.8135 - val_loss: 0.4811 - val_acc: 0.7850
Epoch 80/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4028 - acc: 0.8050 - val_loss: 0.4407 - val_acc: 0.8140
Epoch 81/100
100/100 [==============================] - 20s 197ms/step - loss: 0.4105 - acc: 0.8100 - val_loss: 0.4171 - val_acc: 0.8170
Epoch 82/100
100/100 [==============================] - 20s 198ms/step - loss: 0.3985 - acc: 0.8245 - val_loss: 0.4365 - val_acc: 0.8090
Epoch 83/100
100/100 [==============================] - 20s 199ms/step - loss: 0.4037 - acc: 0.8165 - val_loss: 0.4360 - val_acc: 0.8140
Epoch 84/100
100/100 [==============================] - 20s 198ms/step - loss: 0.3904 - acc: 0.8215 - val_loss: 0.4411 - val_acc: 0.8220
Epoch 85/100
100/100 [==============================] - 20s 199ms/step - loss: 0.3868 - acc: 0.8260 - val_loss: 0.4280 - val_acc: 0.8180
Epoch 86/100
100/100 [==============================] - 20s 203ms/step - loss: 0.3933 - acc: 0.8215 - val_loss: 0.5525 - val_acc: 0.7770
Epoch 87/100
100/100 [==============================] - 20s 200ms/step - loss: 0.3874 - acc: 0.8195 - val_loss: 0.4354 - val_acc: 0.8180
Epoch 88/100
100/100 [==============================] - 20s 195ms/step - loss: 0.3920 - acc: 0.8220 - val_loss: 0.4991 - val_acc: 0.7840
Epoch 89/100
100/100 [==============================] - 20s 196ms/step - loss: 0.3984 - acc: 0.8175 - val_loss: 0.4999 - val_acc: 0.7740
Epoch 90/100
100/100 [==============================] - 20s 197ms/step - loss: 0.3808 - acc: 0.8235 - val_loss: 0.4727 - val_acc: 0.8040
Epoch 91/100
100/100 [==============================] - 20s 197ms/step - loss: 0.3849 - acc: 0.8240 - val_loss: 0.4432 - val_acc: 0.8090
Epoch 92/100
100/100 [==============================] - 20s 197ms/step - loss: 0.3957 - acc: 0.8195 - val_loss: 0.4915 - val_acc: 0.8050
Epoch 93/100
100/100 [==============================] - 19s 191ms/step - loss: 0.3853 - acc: 0.8265 - val_loss: 0.4299 - val_acc: 0.8260
Epoch 94/100
100/100 [==============================] - 18s 176ms/step - loss: 0.3879 - acc: 0.8305 - val_loss: 0.4631 - val_acc: 0.8010
Epoch 95/100
100/100 [==============================] - 18s 177ms/step - loss: 0.3881 - acc: 0.8205 - val_loss: 0.4698 - val_acc: 0.8070
Epoch 96/100
100/100 [==============================] - 18s 176ms/step - loss: 0.3783 - acc: 0.8255 - val_loss: 0.4098 - val_acc: 0.8290
Epoch 97/100
100/100 [==============================] - 18s 178ms/step - loss: 0.3902 - acc: 0.8215 - val_loss: 0.4380 - val_acc: 0.8150
Epoch 98/100
100/100 [==============================] - 18s 177ms/step - loss: 0.3584 - acc: 0.8375 - val_loss: 0.4364 - val_acc: 0.8200
Epoch 99/100
100/100 [==============================] - 18s 178ms/step - loss: 0.3792 - acc: 0.8350 - val_loss: 0.4111 - val_acc: 0.8250
Epoch 100/100
100/100 [==============================] - 18s 177ms/step - loss: 0.3866 - acc: 0.8275 - val_loss: 0.4126 - val_acc: 0.8140
```
#### Output
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.5_data_augment_dropout_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.2.5_data_augment_dropout_2.png?raw=true)

### Train with pretrained weights
Training is very fast, because it only deal with two `Dense` layers - an epoch takes less than one second even on CPU

`$ python dlp_5.3.1_train_with_pretrained_weights.py`

```
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Epoch 1/30
100/100 [==============================] - 1s 10ms/step - loss: 0.6441 - acc: 0.6330 - val_loss: 0.4558 - val_acc: 0.8360
Epoch 2/30
100/100 [==============================] - 1s 9ms/step - loss: 0.4403 - acc: 0.8015 - val_loss: 0.3766 - val_acc: 0.8320
Epoch 3/30
100/100 [==============================] - 1s 9ms/step - loss: 0.3629 - acc: 0.8470 - val_loss: 0.3228 - val_acc: 0.8830
Epoch 4/30
100/100 [==============================] - 1s 9ms/step - loss: 0.3155 - acc: 0.8670 - val_loss: 0.3016 - val_acc: 0.8890
Epoch 5/30
100/100 [==============================] - 1s 9ms/step - loss: 0.2848 - acc: 0.8890 - val_loss: 0.2816 - val_acc: 0.8910
Epoch 6/30
100/100 [==============================] - 1s 9ms/step - loss: 0.2611 - acc: 0.8970 - val_loss: 0.2813 - val_acc: 0.8870
Epoch 7/30
100/100 [==============================] - 1s 9ms/step - loss: 0.2456 - acc: 0.9040 - val_loss: 0.2638 - val_acc: 0.8970
Epoch 8/30
100/100 [==============================] - 1s 9ms/step - loss: 0.2274 - acc: 0.9145 - val_loss: 0.2576 - val_acc: 0.8970
Epoch 9/30
100/100 [==============================] - 1s 9ms/step - loss: 0.2140 - acc: 0.9185 - val_loss: 0.2539 - val_acc: 0.8920
Epoch 10/30
100/100 [==============================] - 1s 9ms/step - loss: 0.2035 - acc: 0.9195 - val_loss: 0.2478 - val_acc: 0.8990
Epoch 11/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1975 - acc: 0.9240 - val_loss: 0.2443 - val_acc: 0.8980
Epoch 12/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1823 - acc: 0.9385 - val_loss: 0.2447 - val_acc: 0.8970
Epoch 13/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1728 - acc: 0.9375 - val_loss: 0.2390 - val_acc: 0.9020
Epoch 14/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1728 - acc: 0.9360 - val_loss: 0.2366 - val_acc: 0.9050
Epoch 15/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1577 - acc: 0.9475 - val_loss: 0.2351 - val_acc: 0.9090
Epoch 16/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1583 - acc: 0.9420 - val_loss: 0.2447 - val_acc: 0.8970
Epoch 17/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1511 - acc: 0.9480 - val_loss: 0.2370 - val_acc: 0.9040
Epoch 18/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1486 - acc: 0.9430 - val_loss: 0.2320 - val_acc: 0.9040
Epoch 19/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1371 - acc: 0.9530 - val_loss: 0.2341 - val_acc: 0.9080
Epoch 20/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1333 - acc: 0.9545 - val_loss: 0.2349 - val_acc: 0.9050
Epoch 21/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1277 - acc: 0.9540 - val_loss: 0.2326 - val_acc: 0.9080
Epoch 22/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1210 - acc: 0.9615 - val_loss: 0.2306 - val_acc: 0.9110
Epoch 23/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1147 - acc: 0.9610 - val_loss: 0.2360 - val_acc: 0.9070
Epoch 24/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1091 - acc: 0.9640 - val_loss: 0.2347 - val_acc: 0.9090
Epoch 25/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1085 - acc: 0.9640 - val_loss: 0.2543 - val_acc: 0.8900
Epoch 26/30
100/100 [==============================] - 1s 9ms/step - loss: 0.1026 - acc: 0.9670 - val_loss: 0.2432 - val_acc: 0.8970
Epoch 27/30
100/100 [==============================] - 1s 9ms/step - loss: 0.0990 - acc: 0.9665 - val_loss: 0.2313 - val_acc: 0.9100
Epoch 28/30
100/100 [==============================] - 1s 9ms/step - loss: 0.0960 - acc: 0.9660 - val_loss: 0.2519 - val_acc: 0.8960
Epoch 29/30
100/100 [==============================] - 1s 9ms/step - loss: 0.0875 - acc: 0.9720 - val_loss: 0.2326 - val_acc: 0.9130
Epoch 30/30
100/100 [==============================] - 1s 8ms/step - loss: 0.0894 - acc: 0.9735 - val_loss: 0.2328 - val_acc: 0.9100
```

#### Output
##### acc: 0.9735, val_cc: 0.9100
It reached a validation accuracy of about 91%.
We used dropout but still we got overfitting almost from the start. 
That's because this technique doesn't use data augmentation.

![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.3.1_train_with_pretrained_weights_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.3.1_train_with_pretrained_weights_2.png?raw=true)


### Train with pretrained weights and plus data augmentation
This technique is much slower and more expensive but allows to use data augmentation during training.

`$ python dlp_5.3.1_train_with_pretrained_weights_and_data_augmentation.py`

#### Output
##### acc: 0.9890, val_cc: 0.9660
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.3.1_train_with_pretrained_weights_and_data_augmentation_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.3.1_train_with_pretrained_weights_and_data_augmentation_2.png?raw=true)


### Train with pretrained weights and plus data augmentation
This technique is much slower and more expensive but allows to use data augmentation during training.

`$ python dlp_5.3.1_train_with_pretrained_weights_with_finetuning.py`

#### Output
##### acc: 0.9860 - val_loss: 0.1408 - val_acc: 0.9680
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.3.1_train_with_pretrained_weights_with_finetuning_1.png?raw=true)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.3.1_train_with_pretrained_weights_with_finetuning_2.png?raw=true)

## dlp 5.4 Visualizing what convnets learn
### dlp 5.4.1 Visualizing intermediate activations
Run `dlp_5.4.1_visualizing_intermediate_activations.py`

```
$ python dlp_5.4.1_visualizing_intermediate_activations.py
```

```
________________________________________________________________
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
dropout (Dropout)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               3211776   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
```
#### Input image. 

inpute shape: (1, 150, 150, 3)
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_1.png?raw=true)


#### Activation of the first convolution layer for the input image:

First activate layer shape: (1, 148, 148, 32)

The fourth channel of the activation of the first layer of the original model:

![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_2.png?raw=true)

The 7th channel of the activation of the first layer of the original model:

![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_3.png?raw=true)

#### Visualization of all the activations in the network:

conv2d:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_4.png?raw=true)

max_pooling2d:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_5.png?raw=true)

conv2d_1:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_6.png?raw=true)

max_pooling2d_1:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_7.png?raw=true)

conv2d_2:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_8.png?raw=true)

max_pooling2d_2:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_9.png?raw=true)

conv2d_3:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_10.png?raw=true)

max_pooling2d_3:
![alt text](https://github.com/martianvenusian/dlp/blob/main/images/dlp_5.4.1_visualizing_intermediate_activations_11.png?raw=true)