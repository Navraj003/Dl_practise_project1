import tensorflow as tf
from keras.layers import Input,Conv2D,MaxPool2D, Flatten,Dense,Dropout
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
train= pd.read_csv('archive/sign_mnist_train.csv')
test= pd.read_csv('archive/sign_mnist_test.csv')


# Preprocess the data
x_train=train.drop('label',axis=1).values
y_train=train['label'].values
x_test=test.drop('label',axis=1).values
y_test=test['label'].values

# Reshape the data
x_train = x_train.reshape(-1, 28, 28, 1)/ 255.0
x_test = x_test.reshape(-1, 28, 28, 1)/ 255.0

print("Unique labels in training set:", np.unique(y_train))
print("Unique labels in test set:", np.unique(y_test))


#one-hot encode the labels
y_train = to_categorical(y_train, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)

#Functional API model
inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(26, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

model.save('model.h5')