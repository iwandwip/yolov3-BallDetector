#!/usr/bin/python

# import the necessary packages
import sys
import os
import rospy
# import rosparam
# from std_msgs.msg import Int8

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
# from kri_2021.msg import CoordinateBall, StateBall
# from kri_2021.msg import KoordinatBola, BolaState

# from geometry_msgs.msg import Point
# from geometry_msgs.msg import Twist

import numpy as np
import cv2
import time

import roslib


class Yolov3:
    def __init__(self):
        self.bridge = CvBridge()
        self.classes = None
        self.models = "models/data.weights"
        self.config = "cfg/config.cfg"

        with open("classes/class.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet(self.models,
                                   self.config)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]
        # fps
        self.fps = 0
        self._prev_time = 0
        self._new_time = 0

    @staticmethod
    def image_resize(image,
                     width=None,
                     height=None,
                     interpolation=cv2.INTER_AREA):
        dim = None
        w = image.shape[1]
        h = image.shape[0]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=interpolation)
        return resized

    @staticmethod
    def detect_obj(outs=None,
                   class_ids=None,
                   confidences=None,
                   boxes=None,
                   width=None,
                   height=None):

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    def callback_image(self, data):
        global frame
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = self.image_resize(image=frame, width=640)
        height, width, channels = frame.shape

        self._new_time = time.time()
        self.fps = 1 / (self._new_time - self._prev_time)
        self._prev_time = self._new_time

        try:
            cv2.putText(frame, str(int(self.fps)) + " fps", (20, 40), 0, 1,
                        [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        except RuntimeError as e:
            print(e)

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                     (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []

        self.detect_obj(outs=outs,
                        class_ids=class_ids,
                        confidences=confidences,
                        boxes=boxes,
                        width=width,
                        height=height)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                if label == "sports_ball":  # bola
                    x_ball, y_ball, w_ball, h_ball = boxes[i]
                    color = self.colors[class_ids[i]]

                    cv2.rectangle(frame, (x_ball, y_ball),
                                  (x_ball + w_ball, y_ball + w_ball),
                                  color, 2)

                    tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
                    c1, c2 = (int(x_ball), int(y_ball)), (int(w_ball), int(h_ball))

                    if label:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

                        cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(frame, label + " " + str(int(confidences[i] * 100)) + "%",
                                    (c1[0], c1[1] - 2), 0,
                                    tl / 3, [225, 255, 255],
                                    thickness=tf, lineType=cv2.LINE_AA)

                        cv2.circle(frame, (int(x_ball + int(w_ball / 2)),
                                           int(y_ball + int(h_ball / 2))), 4, color, -1)

                        cv2.putText(frame, str(int(x_ball + int(w_ball / 2))) + ", " + str(
                            int(y_ball + int(h_ball / 2))),
                                    (int(x_ball + int(w_ball / 2) + 10),
                                     int(y_ball + int(h_ball / 2) + 10)),
                                    font, tl / 2, [255, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)

        cv2.imshow("main frame", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node('processing_image_bola', anonymous=False)
    y = Yolov3()
    try:
        rospy.Subscriber("usb_cam/image_raw", Image, y.callback_image)
        rospy.spin()

    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
