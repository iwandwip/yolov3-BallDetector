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


# import roslib


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

    def giveLabels(self,
                   _frame=None,
                   _labels=None,
                   _indexes=None,
                   _font=None,
                   _boxes=None,
                   _class_ids=None,
                   _confidences=None):

        x_item, y_item, w_item, h_item = 0, 0, 0, 0

        for i in range(len(_boxes)):
            if i in _indexes:
                label = str(self.classes[_class_ids[i]])
                if label == _labels:  # bola
                    x_item, y_item, w_item, h_item = _boxes[i]

                    color = self.colors[_class_ids[i]]
                    cv2.rectangle(_frame,
                                  (x_item, y_item),
                                  (x_item + w_item, y_item + w_item),
                                  color, 2)

                    tl = round(0.002 * (_frame.shape[0] + _frame.shape[1]) / 2) + 1  # line/font thickness
                    c1, c2 = (int(x_item), int(y_item)), (int(w_item), int(h_item))

                    if label:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

                        cv2.rectangle(_frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(_frame, label + " " + str(int(_confidences[i] * 100)) + "%",
                                    (c1[0], c1[1] - 2), 0,
                                    tl / 3, [225, 255, 255],
                                    thickness=tf, lineType=cv2.LINE_AA)

                        cv2.circle(_frame, (int(x_item + int(w_item / 2)),
                                            int(y_item + int(h_item / 2))), 4, color, -1)

                        cv2.putText(_frame, str(int(x_item + int(w_item / 2))) + ", " + str(
                            int(y_item + int(h_item / 2))),
                                    (int(x_item + int(w_item / 2) + 10),
                                     int(y_item + int(h_item / 2) + 10)),
                                    _font, tl / 2, [255, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)

        return _frame, x_item, y_item, w_item, h_item

    def callback_image(self, data):
        frame = [[[]]]
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        frame = self.image_resize(image=frame, width=450)
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

        frame_bola, x_ball, y_ball, w_ball, h_ball = self.giveLabels(
            _frame=frame,
            _labels="sports_ball",
            _indexes=indexes,
            _font=font,
            _boxes=boxes,
            _class_ids=class_ids,
            _confidences=confidences)

        # frame_gawang = self.giveLabels(_frame=frame,
        #                                _labels="goals",
        #                                _indexes=indexes,
        #                                _font=font,
        #                                _boxes=boxes,
        #                                _class_ids=class_ids,
        #                                _confidences=confidences)

        cv2.imshow("frame_bola", frame_bola)
        # cv2.imshow("frame_gawang", frame_gawang)
        cv2.waitKey(1)


class RosHandler:
    _kf_FocalCamera = 347.10111738231086
    _kf_BallWidth = 14.6

    def __init__(self,
                 _x_pos, _y_pos,
                 _w_box, _h_box,
                 _width_frame, _height_frame):
        self._x_position = _x_pos
        self._y_position = _y_pos
        self._w_box = _w_box
        self._h_box = _h_box
        self._width_frame = _width_frame
        self._height_frame = _height_frame

    def getXfilter(self):
        return (self._x_position /
                self._width_frame) * 2 - 1

    def getYFilter(self):
        return (self._y_position /
                self._height_frame) * 2 - 1

    def sendBallPosition(self):
        pass

    def sendGoalsPosition(self):
        pass

    def sendBallState(self):
        pass

    def sendGoalsState(self):
        pass

    def getDistance(self):
        return (self._kf_BallWidth * self._kf_FocalCamera) / (
                (self._w_box + self._h_box) / 2)

    def getFocal(self):
        return self._kf_BallWidth * (
                (self._w_box + self._h_box) / 2) / 64.2  # real obj dist


if __name__ == "__main__":
    rospy.init_node('processing_image_bola', anonymous=False)
    y = Yolov3()
    try:
        rospy.Subscriber("usb_cam/image_raw", Image, y.callback_image)
        rospy.spin()

    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()
        pass
