### CNN-mnist-enhance

調整參數使用兩層conv2D以及pooling 嘗試優化結果

```python
# 導入函式庫
import numpy as np  
import keras
# from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
from matplotlib import pyplot as plt
import os

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 建立簡單的線性執行的模型
model = keras.models.Sequential()


model.add(Conv2D(filters=24, kernel_size=(5, 5),
                 activation='relu', padding='same',
                 input_shape=(28, 28, 1))) # 28 * 28 * 1( 單色階)
model.add(MaxPooling2D(pool_size=(2, 2))) #

# ((shape of width of the filter * shape of height of the filter * number of filters in the previous layer+1)*number of filters)
model.add(Conv2D(filters=48, kernel_size=(5, 5),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 

model.add(Conv2D(filters=64, kernel_size=(5, 5),
                 activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2))) # 

model.add(Flatten())

# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=256,  activation='relu')) 
# Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1

y_TestOneHot = np_utils.to_categorical(y_test) 

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000001000，即第7個值為 1
y_TrainOneHot = np_utils.to_categorical(y_train) 
y_TestOneHot = np_utils.to_categorical(y_test) 


# 將 training 的 input 資料轉為2維
X_train_2D = X_train.reshape(60000, 28, 28, 1).astype('float32')  
X_test_2D = X_test.reshape(10000, 28, 28, 1).astype('float32')  

x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255

print(model.summary())
```

可以使用color函數將RGB轉為GRAY,也可進行黑白轉換

```python
import matplotlib.pyplot as plt
from skimage import color
test = color.rgb2gray(plt.imread('手寫.png')) # 手寫.png 6.jpg
plt.figure(figsize=(2,2))
plt.imshow(test, cmap='gray')
plt.show()
```

```python
test.shape # 需要轉換成 (1張, 28上下, 28左右, 1單色階)
test.min(), test.max()
(0.0, 1.0)

# ValueError: Error when checking input: expected dense_1_input to have shape (784,) but got array with shape (1,)
import numpy as np
print(model.predict(test.reshape(1, 28, 28, 1)) )
print(np.argmax(model.predict(test.reshape(1, 28, 28, 1))))
```

小貓小狗使用vgg16
