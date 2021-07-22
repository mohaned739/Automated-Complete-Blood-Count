import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import os

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', './data/custom-yolov4-detector_final.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/BloodImage_00339.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')


counted_classes = 0

def main(_argv):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = FLAGS.size
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]

    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, FLAGS.weights)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, FLAGS.weights)
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights(model, FLAGS.weights)

        pred_bbox = model.predict(image_data)
    else:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # print(input_details)
        # print(output_details)
        interpreter.set_tensor(input_details[0]['index'], image_data)
        interpreter.invoke()
        pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    if FLAGS.model == 'yolov4':
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
    else:
        pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = utils.nms(bboxes, 0.213, method='nms')
    global counted_classes
    counted_classes = utils.Count_Cells(bboxes)

    image = utils.draw_bbox(original_image, bboxes)
    image = Image.fromarray(image)

    root = Tk()
    lbl = Label(root)
    lbl.grid(row=0,column=0)

    image.thumbnail((480, 400))
    img = ImageTk.PhotoImage(image)
    label=tk.Label(image=img)
    label.image=img
    label.grid(row=0,column=0)

    frame=Frame(root,width=500,height=350,pady=3)
    frame.grid(row=3,column=0)
    e = Entry(frame, width=8, fg='red',font=('Arial',16,'bold'))
    e.grid(row=3, column=0)
    e.insert(END, 'Platlets')

    e = Entry(frame, width=5, fg='blue',font=('Arial',16,'bold'))
    e.grid(row=3, column=1)
    e.insert(END, 'RBCs')

    e = Entry(frame, width=5, fg='green', font=('Arial', 16, 'bold'))
    e.grid(row=3, column=2)
    e.insert(END, 'WBCs')

    e = Entry(frame, width=8, fg='red', font=('Arial', 16, 'bold'))
    e.grid(row=4, column=0)
    try:
        e.insert(END, '%d' % (counted_classes['Platelets']))
    except:
        e.insert(END, '0')
    e = Entry(frame, width=5, fg='blue', font=('Arial', 16, 'bold'))
    e.grid(row=4, column=1)
    try:
        e.insert(END, '%d' % (counted_classes['RBC']))
    except:
        e.insert(END, '0')
    e = Entry(frame, width=5, fg='green', font=('Arial', 16, 'bold'))
    e.grid(row=4, column=2)
    try:
        e.insert(END, '%d' % (counted_classes['WBC']))
    except:
        e.insert(END, '0')

    root.title("Report")
    root.geometry("500x500")
    root.mainloop()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
