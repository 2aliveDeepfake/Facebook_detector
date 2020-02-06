#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


def load_face_model():
    # Path to frozen detection graph. This is the actual face_model that is used for the object detection.
    PATH_TO_CKPT = './face_model/frozen_inference_graph_face.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './face_label/face_label_map.pbtxt'

    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
      (im_width, im_height) = image.size
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

    # out = None
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)

    return sess, detection_graph, category_index


def detect_face (PATH_TO_VIDEO, number,count, f_sess, f_detection_graph, f_category_index) :
    code_start = time.time()

    num = number
    c = count
    cap = cv2.VideoCapture(PATH_TO_VIDEO)
    face_list = []
    frame_num = 300

    while frame_num:
        frame_num -= 1
        ret, image = cap.read()
        if ret == 0:
            break
        if (int(cap.get(1)) % num == c):
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the face_model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = f_detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = f_detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = f_detection_graph.get_tensor_by_name('detection_scores:0')
            classes = f_detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = f_detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = f_sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # ========================================================
            # Visualization of the results of a detection.
            left, right, top, bottom = vis_util.visualize_boxes_and_labels_on_image_array(
                #          image_np,
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                f_category_index,
                use_normalized_coordinates=True,
                line_thickness=4,
                min_score_thresh=0.3)

            result = [left, right, top, bottom]
            face_list += [result]

    load_face_model_time = time.time() - code_start

    return face_list, load_face_model_time
