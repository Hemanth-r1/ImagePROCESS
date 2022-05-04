import numpy as np
import cv2
import pickle

width = 640
height = 480
threshold = 0.65

cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

pickle_in = open('model_trained_10.p', 'rb')
model = pickle.load(pickle_in)
# model = pickle.load('model_trained_10.p')


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255
    return image


while True:
    success, imageOriginal = cap.read()
    image = np.array(imageOriginal)
    image = cv2.resize(image, (32, 32))
    image = preprocessing(image)
    cv2.imshow("Processed image", image)
    image = image.reshape(1, 32, 32, 1)

    # predict
    classIndex = int(model.predict_classes(image))
    print(classIndex)
    prediction = model.predict(image)
    print(prediction)
    probVal = np.amax(prediction)
    print(classIndex, probVal)

    if probVal > threshold:
        cv2.putText(imageOriginal, str(classIndex) + "  " + str(probVal),(50,50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imageOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
