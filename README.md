# Sound-Signal-Processing-for-Deep-Learning-using-Python-Custom-Datasets
 In this project you will learn how to prepare and process your own custom audio dataset for Deep Learning Training and Test operations.
 No raw audio files found. Generating synthetic audio files...
Synthetic audio files generated.
MFCC feature shape: (15, 40, 174)
Labels shape: (15,)
 ![image](https://github.com/user-attachments/assets/25fdb25a-1853-4784-bd38-5cd2f2490b94)
 /usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
# Model: "sequential"
Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 38, 172, 32)         │             320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 19, 86, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 17, 84, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 8, 42, 64)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 21504)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 125)                 │       2,688,125 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 250)                 │          31,500 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 125)                 │          31,375 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 3)                   │             378 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 2,770,194 (10.57 MB)
 Trainable params: 2,770,194 (10.57 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step - accuracy: 0.3333 - loss: 1.3104 - val_accuracy: 0.3333 - val_loss: 1.8822
Epoch 2/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 927ms/step - accuracy: 0.3333 - loss: 1.8595 - val_accuracy: 0.6667 - val_loss: 2.5799
Epoch 3/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 454ms/step - accuracy: 0.6667 - loss: 2.5386 - val_accuracy: 0.6667 - val_loss: 0.4847
Epoch 4/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 268ms/step - accuracy: 0.6667 - loss: 0.4222 - val_accuracy: 1.0000 - val_loss: 0.0120
Epoch 5/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 267ms/step - accuracy: 1.0000 - loss: 0.0108 - val_accuracy: 0.6667 - val_loss: 0.2641
Epoch 6/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 242ms/step - accuracy: 0.8333 - loss: 0.2295 - val_accuracy: 1.0000 - val_loss: 0.0150
Epoch 7/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 240ms/step - accuracy: 1.0000 - loss: 0.0119 - val_accuracy: 1.0000 - val_loss: 9.8671e-04
Epoch 8/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 225ms/step - accuracy: 1.0000 - loss: 7.9500e-04 - val_accuracy: 1.0000 - val_loss: 2.6280e-04
Epoch 9/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 277ms/step - accuracy: 1.0000 - loss: 2.1177e-04 - val_accuracy: 1.0000 - val_loss: 7.9824e-05
Epoch 10/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 304ms/step - accuracy: 1.0000 - loss: 6.2571e-05 - val_accuracy: 1.0000 - val_loss: 3.4013e-05
![image](https://github.com/user-attachments/assets/93fdac5a-403f-4c2f-b7f2-4879c81b76bd)


