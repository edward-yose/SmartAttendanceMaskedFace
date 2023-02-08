import threading
import tkinter

import cv2
import os
import time
import math
import datetime
import numpy as np

from face_alignment import FaceMaskDetection
from attendance import DailyAttendanceEntry
from tools import model_restore_from_pb

import tkinter as tk
from tkinter import messagebox
from threading import *

import tensorflow

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
print("[GUI Application]Tensorflow version: ", tf.__version__)

img_format = {'png', 'jpg', 'bmp'}
room_id = "001"


def video_init(camera_source=0, resolution="480", to_write=False, save_dir=None):
    print("[INFO] : High Level Function Video Init Triggered ")

    # ----var
    writer = None
    resolution_dict = {"480": [480, 640], "720": [720, 1280], "1080": [1080, 1920]}

    # ----camera source connection
    cap = cv2.VideoCapture(camera_source)

    # ----resolution decision
    if resolution in resolution_dict.keys():
        width = resolution_dict[resolution][1]
        height = resolution_dict[resolution][0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if to_write is True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_path = 'demo.avi'
        if save_dir is not None:
            save_path = os.path.join(save_dir, save_path)
        writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

    return cap, height, width, writer


class UIMainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry("1560x720")
        self.title("Smart Attendance Management System")

        self.videoThread = threading.Thread(target=VideoThread)
        self.videoThread.start()


class VideoThread:
    # gvar
    camera_source = 0  # can be also the path of a clip
    pb_path = "models/weights_15.pb"
    node_dict = {'input': 'input:0',
                 'keep_prob': 'keep_prob:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 }

    ref_dir = r"imageDatabase"

    # var
    frame_count = 0
    FPS = "loading"
    face_mask_model_path = r'models/face_mask_detection.pb'
    margin = 40
    id2class = {0: 'Mask', 1: 'NoMask'}
    batch_size = 32
    threshold = 0.8

    def run(self):
        self.ThreadActive = True

        # Video streaming init
        cap, height, width, writer = video_init(camera_source=self.camera_source,
                                                resolution="720",
                                                to_write=False,
                                                save_dir=None)

        # Face detection init (GPU RATION 0.1)
        fmd = FaceMaskDetection(self.face_mask_model_path, self.margin, GPU_ratio=None)

        # Face recognition init
        sess, tf_dict = model_restore_from_pb(self.pb_path, self.node_dict, GPU_ratio=None)
        print("[INFO] TF DICT : ", tf_dict)
        tf_input = tf_dict['input']
        tf_phase_train = tf_dict['phase_train']
        tf_embeddings = tf_dict['embeddings']
        model_shape = tf_input.shape.as_list()
        print("[INFO] The mode shape of face recognition:", model_shape)
        print("[INFO] TF DICT key list : ", tf_dict.keys())
        feed_dict = {tf_phase_train: False}
        if 'keep_prob' in tf_dict.keys():
            tf_keep_prob = tf_dict['keep_prob']
            feed_dict[tf_keep_prob] = 1.0

            # ----read images from the database
            d_t = time.time()
            paths = [file.path for file in os.scandir(self.ref_dir) if file.name[-3:] in img_format]
            len_ref_path = len(paths)
            if len_ref_path == 0:
                print("No images in ", self.ref_dir)
            else:
                ites = math.ceil(len_ref_path / self.batch_size)
                embeddings_ref = np.zeros([len_ref_path, tf_embeddings.shape[-1]], dtype=np.float32)

                for i in range(ites):
                    num_start = i * self.batch_size
                    num_end = np.minimum(num_start + self.batch_size, len_ref_path)

                    batch_data_dim = [num_end - num_start]
                    batch_data_dim.extend(model_shape[1:])
                    batch_data = np.zeros(batch_data_dim, dtype=np.float32)

                    for idx, path in enumerate(paths[num_start:num_end]):
                        img = cv2.imread(path)
                        if img is None:
                            print("read failed:", path)
                        else:
                            # cv2.imshow(img)
                            print(img.shape, (type(model_shape[2]), model_shape[1]))
                            img = cv2.resize(img, (int(model_shape[2]), int(model_shape[1])))
                            img = img[:, :, ::-1]  # change the color format
                            batch_data[idx] = img
                    batch_data /= 255
                    feed_dict[tf_input] = batch_data

                    embeddings_ref[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)

                d_t = time.time() - d_t

                print("ref embedding shape", embeddings_ref.shape)
                print("It takes {} secs to get {} embeddings".format(d_t, len_ref_path))

            # ----tf setting for calculating distance
            if len_ref_path > 0:
                with tf.Graph().as_default():
                    tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
                    tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
                    tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
                    # ----GPU setting
                    config = tf.ConfigProto(log_device_placement=True,
                                            allow_soft_placement=True,
                                            )
                    config.gpu_options.allow_growth = True
                    sess_cal = tf.Session(config=config)
                    sess_cal.run(tf.global_variables_initializer())

                feed_dict_2 = {tf_ref: embeddings_ref}

        while self.ThreadActive:
            ret, img = cap.read()
            if ret:
                # ----image processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = img_rgb.astype(np.float32)
                img_rgb /= 255
                # ----face detection
                img_fd = cv2.resize(img_rgb, fmd.img_size)
                img_fd = np.expand_dims(img_fd, axis=0)
                bboxes, re_confidence, re_classes, re_mask_id = fmd.inference(img_fd, height, width)
                if len(bboxes) > 0:
                    for num, bbox in enumerate(bboxes):
                        class_id = re_mask_id[num]
                        if class_id == 0:
                            color = (0, 255, 0)  # (B,G,R) --> Green(with masks)
                        else:
                            color = (0, 0, 255)  # (B,G,R) --> Red(without masks)
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                        # cv2.putText(img, "%s: %.2f" % (self.id2class[class_id], re_confidence[num]), (bbox[0] + 2, bbox[1] - 2),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

                        # ----face recognition
                        name = ""
                        if len_ref_path > 0:
                            img_fr = img_rgb[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]  # crop
                            img_fr = cv2.resize(img_fr, (int(model_shape[2]), int(model_shape[1])))  # resize
                            img_fr = np.expand_dims(img_fr, axis=0)  # make 4 dimensions

                            feed_dict[tf_input] = img_fr
                            embeddings_tar = sess.run(tf_embeddings, feed_dict=feed_dict)
                            feed_dict_2[tf_tar] = embeddings_tar[0]
                            distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                            arg = np.argmin(distance)  # index of the smallest distance

                            if distance[arg] < self.threshold:
                                name = paths[arg].split("\\")[-1].split(".")[0]  # --> get name

                        cv2.putText(img,
                                    "{} ({}%)".format(name, format(re_confidence[num] * 100, '.2f')),
                                    (bbox[0] + 4, bbox[1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                        cv2.putText(img, "{}".format(self.id2class[class_id]),
                                    (bbox[0] + 4, bbox[1] - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        DailyAttendanceEntry.mark_this_day_attendance(name, room_id)

                if self.frame_count == 0:
                    t_start = time.time()
                self.frame_count += 1
                if self.frame_count >= 10:
                    self.FPS = "FPS=%1f" % (10 / (time.time() - t_start))
                    self.frame_count = 0
                print("FPS: ", self.FPS)

                cv2.putText(img, self.FPS, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                Image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(Image.data,
                                           Image.shape[1],
                                           Image.shape[0],
                                           QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
                # TODO Implement FPS Catch Report
                self.ImageUpdate.emit(Pic)

                if writer is not None:
                    writer.write(img)

        if writer is not None:
            writer.release()

    def stop(self):
        self.ThreadActive = False
        self.quit()
