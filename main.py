import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from keras.legacy_tf_layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizer_v1 import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

path = 'myData'
testRatio = 0.2
validationRatio = 0.2
imageDimensions = (32, 32, 3)

batchSizeVal = 50
epochVal = 10
stepsForEpochVal = 200

images = []
classNo = []
myList = os.listdir(path)
print("total number of class detected", len(myList))
noOfClasses = len(myList)
print("Importing classes")
for count in range(0, noOfClasses):  # to iterate through the directory
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")  # to show it in line
    count += 1
print("")
print("Total Images in Images list = ", len(images))
print("Total IDS in classNo List = ", len(classNo))

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)

# splitting the data

X_train, X_test, Y_train, Y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validationRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    # print(len(np.where(Y_train == x) [0]))
    numOfSamples.append(len(np.where(Y_train == x)[0]))

print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


# print(X_train[30].shape) # to compare before and after pre processing

def preprocessing(imageP):
    imageP = cv2.cvtColor(imageP, cv2.COLOR_BGR2GRAY)
    imageP = cv2.equalizeHist(imageP)
    imageP = imageP / 255

    return imageP


# to check if its working
# img = preprocessing(X_train[30])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))

# print(X_train[30].shape)
# img = preprocessing(X_train[30])
# img = cv2.resize(img, (300, 300))
# cv2.imshow("PreProcessed", img)
# cv2.waitKey(0)

# print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# print(X_train.shape)

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
Y_train = to_categorical(Y_train, noOfClasses)
Y_test = to_categorical(Y_test, noOfClasses)
Y_validation = to_categorical(Y_validation, noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = myModel()
print((model.summary()))

history = model.fit_generator(dataGen.flow(X_train, Y_train,
                                           batch_size=batchSizeVal), steps_per_epoch=stepsForEpochVal,
                              epochs=epochVal, validation_data=(X_validation, Y_validation), shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')

plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)
print("test score = ", score[0])
print('Test accuracy = ', score[1])

pickle_out = open('model_trained_10.p', 'wb')  # write bytes
pickle.dump(model, pickle_out)
pickle_out.close()
