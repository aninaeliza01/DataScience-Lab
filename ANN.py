from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)
y_test=to_categorical(y_test)
y_train=to_categorical(y_train)
model=Sequential([Dense(128,activation='relu'),Dense(64,activation='relu'),Dense(10,activation='softmax')])
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5)
loss,accuracy=model.evaluate(x_test,y_test)
print(f'{accuracy}')