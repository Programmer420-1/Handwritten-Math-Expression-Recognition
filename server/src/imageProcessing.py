import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import imutils
from imutils.contours import sort_contours

savedWeight = "LSTM99_v2.h5"
width = 45
height = 45
target_size = (width, height)
channel = 1
batch_size = 64
label_list = ['(',
              ')',
              '+',
              '-',
              '1',
              '2',
              '3',
              '4',
              '5',
              '6',
              '7',
              '8',
              '9',
              '=',
              'div',
              'X']


# def bBoxCoordList(inputImage):
#     cnts, gray = preprocessImage(inputImage)
#     chars = []
#     # loop over the contours
#     for c in cnts:
#         # compute the bounding box of the contour
#         (x, y, w, h) = cv2.boundingRect(c)
#         # filter out bounding boxes, ensuring they are neither too small
#         # nor too large
#         if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
#             # extract the character and threshold it to make the character
#             # appear as *white* (foreground) on a *black* background, then
#             # grab the width and height of the thresholded image
#             roi = gray[y:y + h, x:x + w]
#             thresh = cv2.threshold(roi, 0, 255,
#                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#             (tH, tW) = thresh.shape
#             # if the width is greater than the height, resize along the
#             # width dimension
#             if tW > tH:
#                 thresh = imutils.resize(thresh, width=45)
#             # otherwise, resize along the height
#             else:
#                 thresh = imutils.resize(thresh, height=45)

#             # re-grab the image dimensions (now that its been resized)
#             # and then determine how much we need to pad the width and
#             # height such that our image will be 32x32
#             (tH, tW) = thresh.shape
#             dX = int(max(0, 45 - tW) / 2.0)
#             dY = int(max(0, 45 - tH) / 2.0)
#             # pad the image and force 32x32 dimensions
#             padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
#                                         left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
#                                         value=(255, 255, 255))
#             padded = cv2.resize(padded, (45, 45))
#             cv2.imshow("Padded", padded)
#             # prepare the padded image for classification via our
#             # handwriting OCR model
#             # padded = padded.astype("float32") / 255.0
#             padded = np.expand_dims(padded, axis=-1)
#             # update our list of characters that will be OCR'd
#             chars.append((padded, (x, y, w, h)))

#         # small images
#         else:
#             roi = gray[y:y + h, x:x + w]
#             thresh = cv2.threshold(roi, 0, 255,
#                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#             (tH, tW) = thresh.shape
#             dX = int(max(0, 45 - tW) / 2.0)
#             dY = int(max(0, 45 - tH) / 2.0)
#             # pad the image and force 32x32 dimensions
#             padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
#                                         left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
#                                         value=(255, 255, 255))
#             padded = cv2.resize(padded, (45, 45))
#             padded = np.expand_dims(padded, axis=-1)

#             chars.append((padded, (x, y, w, h)))

#     return chars


# def preprocessImage(inputImage):
#     gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)  # Gray scaling
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # remove noise

#     # perform edge detection, find contours in the edge map, and sort the
#     # resulting contours from left-to-right
#     edged = cv2.Canny(blurred, 30, 150)
#     cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_SIMPLE)
#     cnts = imutils.grab_contours(cnts)
#     cnts = sort_contours(cnts, method="left-to-right")[0]

#     return cnts, gray


def pipeline(imageIn):
    inputImage = imageIn.copy()
    # Gray scaling
    gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Set the adaptive thresholding (gasussian) parameters:
    windowSize = 7
    windowConstant = -1

    # Apply the threshold:
    binaryImage = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant)

    # cv2.imshow("Step 1", binaryImage)
    # cv2.waitKey(0)

    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(binaryImage)

    # Set the minimum pixels for the area filter:
    minArea = 10

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(
        1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(
        np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    # cv2.imshow("Step 2", filteredImage)
    # cv2.waitKey(0)

    # Set kernel (structuring element) size:
    kernelSize = 3

    # Set operation iterations:
    opIterations = 1

    # Get the structuring element:
    maxKernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernelSize, kernelSize))

    # Perform closing:
    closingImage = cv2.morphologyEx(
        filteredImage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    # cv2.imshow("Step 3", closingImage)
    # cv2.waitKey(0)

    # Get each bounding box
    # Find the big contours/blobs on the filtered image:
    contours, hierarchy = cv2.findContours(
        closingImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    # The Bounding Rectangles will be stored here:
    boundRect = []
    chars = []

    # Alright, just look for the outer bounding boxes:
    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
    
    # Sort the contours from left to right
    contours_poly = sort_contours(contours_poly, method="left-to-right")[0]


    # Get the bounding box
    for c in contours_poly:
            boundRect.append(cv2.boundingRect(c))

    # Draw the bounding boxes on the (copied) input image:
    for i in range(len(boundRect)):
        color = (0, 255, 0)
        x = int(boundRect[i][0])
        y = int(boundRect[i][1])
        w = int(boundRect[i][2])
        h = int(boundRect[i][3])

        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=45)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=45)

            # re-grab the image dimensions (now that its been resized)
            # and then determine how much we need to pad the width and
            # height such that our image will be 32x32
            (tH, tW) = thresh.shape
            dX = int(max(0, 45 - tW) / 2.0)
            dY = int(max(0, 45 - tH) / 2.0)
            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(255, 255, 255))
            padded = cv2.resize(padded, (45, 45))
            
            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))
            cv2.rectangle(imageIn, (x, y), (x + w, y + h), color, 2)
    
    return runPrediction(chars,imageIn)

def runPrediction(chars, inputImage):
    pred_text = ""
    pred_confidence = ""
    # get the trained model
    input_dim = list(target_size)
    input_dim.append(channel)
    output_n = len(label_list)
    model = getModel(input_dim, output_n)
    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # find the index of the label with the largest corresponding
        # probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = label_list[i]
        pred_text += label + " "
        pred_confidence += "[INFO] {} - {:.2f}%".format(label, prob * 100) + "\n"
        cv2.putText(inputImage, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    expression = pred_text + "\n" + pred_confidence
    return expression, inputImage


def getModel(input_shape, output_n):
    model = Sequential()
    model.add(layers.Conv2D(
        input_shape=input_shape,
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu"))
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu"))
    model.add(layers.MaxPool2D(pool_size=1))

    model.add(layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu"))
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu"))
    model.add(layers.MaxPool2D(pool_size=2))

    model.add(layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu"))
    model.add(layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu"))
    model.add(layers.MaxPool2D(pool_size=2))

    model.add(layers.ConvLSTM1D(
        filters=256,
        kernel_size=3,
        strides=3,
        padding="same",
        activation="relu"
    ))

    model.add(layers.Flatten())
    model.add(layers.Dense(output_n, activation="softmax"))

    model.compile(optimizer="Adam",
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.load_weights(savedWeight)
    return model


