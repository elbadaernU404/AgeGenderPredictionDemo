# Import required modules
#import math
import time
import argparse

import cv2

def getFaceBox(net, frame, threshold_conf=0.7):
    bounding_OpencvDnn = frame.copy()
    bounding_Height = bounding_OpencvDnn.shape[0]
    bounding_Width = bounding_OpencvDnn.shape[1]
    blobs = cv2.dnn.blobFromImage(bounding_OpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blobs)
    detections = net.forward()
    bounding_bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold_conf:
            x1 = int(detections[0, 0, i, 3] * bounding_Width)
            y1 = int(detections[0, 0, i, 4] * bounding_Height)
            x2 = int(detections[0, 0, i, 5] * bounding_Width)
            y2 = int(detections[0, 0, i, 6] * bounding_Height)
            bounding_bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(bounding_OpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(bounding_Height/150)), 8)
    return bounding_OpencvDnn, bounding_bboxes

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
#genderList.decode("utf-8")

# Load network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream
cap = cv2.VideoCapture(args.input if args.input else 0)
padding = 20
while cv2.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv2.waitKey()
        break

    frameFace, bounding_bboxes = getFaceBox(faceNet, frame)
    if not bounding_bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bounding_bbox in bounding_bboxes:
        print("=====================================Face Found=====================================")
        # print(bounding_bbox)
        face = frame[max(0,bounding_bbox[1]-padding):min(bounding_bbox[3]+padding,frame.shape[0]-1),max(0,bounding_bbox[0]-padding):min(bounding_bbox[2]+padding, frame.shape[1]-1)]

        blobs = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blobs)
        genderPredction = genderNet.forward()
        gender = genderList[genderPredction[0].argmax()]
        # print("Gender Output : {}".format(genderPredction))

        print("Gender : {}, conf = {:.3f}".format(gender, genderPredction[0].max()))

        ageNet.setInput(blobs)
        agepredction = ageNet.forward()
        age = ageList[agepredction[0].argmax()]

        print("Age Output : {}".format(agepredction))
        print("Age : {}, conf = {:.3f}".format(age, agepredction[0].max()))

        label = "{},{}".format(gender, age)
        cv2.putText(frameFace, label, (bounding_bbox[0], bounding_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Age & Gender Prediction Demo", frameFace)
        # cv2.imwrite("age-gender-out-{}".format(args.input),frameFace)

    print("time : {:.3f}".format(time.time() - t))
    print("=====================================Round Over=====================================")