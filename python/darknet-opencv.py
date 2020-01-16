from cv2 import dnn, getTextSize, FONT_HERSHEY_SIMPLEX, rectangle, FILLED, putText, VideoCapture
from numpy import argmax, array
import cv2
from PIL import Image
import os
import numpy as np


class yolo3():
    def __init__(self, hasGPU, classesFile, modelConfiguration, modelWeights):
        # Initialize the parameters
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold
        self.inpWidth = 416  # Width of network's input image
        self.inpHeight = 416  # Height of network's input image
        self.frame = None

        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.net = dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
        if hasGPU:
            self.net.setPreferableTarget(dnn.DNN_TARGET_OPENCL)
        else:
            self.net.setPreferableTarget(dnn.DNN_TARGET_CPU)

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawPred(self, classId, conf, left, top, right, bottom, txt_tag):
        # Draw a bounding box.
        rectangle(self.frame, (left, top), (right, bottom), (0, 255, 255), 10)

        if txt_tag:
            label = '%.2f' % conf

            # Get the label for the class name and its confidence
            if self.classes:
                assert (classId < len(self.classes))
                label = '%s:%s' % (self.classes[classId], label)

            # Display the label at the top of the bounding box
            labelSize, baseLine = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            rectangle(self.frame, (left, top - round(1.5 * labelSize[1])),
                      (left + round(1.5 * labelSize[0]), top + baseLine),
                      (255, 255, 255), FILLED)
            putText(self.frame, label, (left, top), FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, outs, txt_tag):
        frameHeight = self.frame.shape[0]
        frameWidth = self.frame.shape[1]
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        results = []
        indices = dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            results.append((self.classes[classIds[i]], confidences[i],
                            (int(left + width / 2), int(top + height / 2), width, height)))
            self.drawPred(classIds[i], confidences[i], left, top, left + width, top + height, txt_tag)
        return sorted(results, key=lambda item: -item[1])

    def performDetect(self, imagePath, txt_tag=True):
        # Open the image file
        # cap = VideoCapture(imagePath)
        # get frame from the video
        # hasFrame, self.frame = cap.read()
        # self.frame = Image.open(imagePath)
        # print(imagePath)
        self.frame = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)
        # cv2.imshow('', self.frame)
        # cv2.waitKey()
        # Create a 4D blob from a frame.
        blob = dnn.blobFromImage(self.frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)
        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.getOutputsNames())
        # Remove the bounding boxes with low confidence
        detections = self.postprocess(outs, txt_tag)
        result = {
            "detections": detections,
            "image": self.frame
        }
        # cv2.imwrite('detect/' + imagePath[imagePath.rindex('\\'):], self.frame)
        return result


if __name__ == '__main__':
    yolo = yolo3(hasGPU=False, classesFile='data/daozha/cfg/voc.names',
                 modelConfiguration='data/daozha/cfg/yolov3-voc.cfg',
                 modelWeights='backup/daozha/yolov3-voc_final.weights')
    path = r'data/daozha/eval_imgs'
    out_path = 'data/result'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for filename in os.listdir(path):
        full_name = os.path.join(path, filename)
        result = yolo.performDetect(full_name)
        cv2.imwrite(os.path.join(out_path, filename), result['image'])
