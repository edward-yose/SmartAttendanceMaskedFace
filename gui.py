import cv2
import os
import time
import math
import datetime
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from face_alignment import FaceMaskDetection
from attendance import DailyAttendanceEntry
from tools import model_restore_from_pb
from report_tools import GenerateReport

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


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1560, 720)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_camera = QtWidgets.QLabel(self.centralwidget)
        self.label_camera.setGeometry(QtCore.QRect(-10, 0, 1278, 720))
        self.label_camera.setMinimumSize(QtCore.QSize(4, 0))
        self.label_camera.setMaximumSize(QtCore.QSize(1280, 720))
        self.label_camera.setText("")
        self.label_camera.setObjectName("label_camera")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(1300, 10, 251, 671))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.datetimeLayout = QtWidgets.QGridLayout()
        self.datetimeLayout.setObjectName("datetimeLayout")
        self.label_date = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_date.setFont(font)
        self.label_date.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_date.setIndent(5)
        self.label_date.setObjectName("label_date")
        self.datetimeLayout.addWidget(self.label_date, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setIndent(5)
        self.label_2.setObjectName("label_2")
        self.datetimeLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_time = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_time.setFont(font)
        self.label_time.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_time.setIndent(5)
        self.label_time.setObjectName("label_time")
        self.datetimeLayout.addWidget(self.label_time, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setIndent(5)
        self.label.setObjectName("label")
        self.datetimeLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setIndent(20)
        self.label_3.setObjectName("label_3")
        self.datetimeLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_room_id = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.label_room_id.setFont(font)
        self.label_room_id.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_room_id.setIndent(20)
        self.label_room_id.setObjectName("label_room_id")
        self.datetimeLayout.addWidget(self.label_room_id, 2, 1, 1, 1)
        self.verticalLayout.addLayout(self.datetimeLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label_info = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.label_info.setFont(font)
        self.label_info.setAlignment(QtCore.Qt.AlignCenter)
        self.label_info.setObjectName("label_info")
        self.verticalLayout.addWidget(self.label_info)
        self.menuLayout = QtWidgets.QVBoxLayout()
        self.menuLayout.setObjectName("menuLayout")
        self.btn_settings = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.btn_settings.setFont(font)
        self.btn_settings.setCheckable(False)
        self.btn_settings.setObjectName("btn_settings")
        self.menuLayout.addWidget(self.btn_settings)
        self.btn_report = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.btn_report.setFont(font)
        self.btn_report.setCheckable(False)
        self.btn_report.setObjectName("btn_report")
        self.menuLayout.addWidget(self.btn_report)
        self.btn_save_new = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.btn_save_new.setFont(font)
        self.btn_save_new.setCheckable(False)
        self.btn_save_new.setObjectName("btn_save_new")
        self.menuLayout.addWidget(self.btn_save_new)
        self.btn_exit = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.btn_exit.setFont(font)
        self.btn_exit.setCheckable(False)
        self.btn_exit.setObjectName("btn_exit")
        self.menuLayout.addWidget(self.btn_exit)
        self.verticalLayout.addLayout(self.menuLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Init Labels
        grey = QPixmap(1280, 720)
        grey.fill(QColor('darkGray'))
        self.label_room_id.setText(room_id)
        self.label_camera.setPixmap(grey)
        self.btn_exit.setDisabled(True)
        self.btn_settings.setDisabled(True)
        self.btn_report.setDisabled(True)
        self.btn_save_new.setDisabled(True)

        # Manipulate datetime attendnace
        now = QDate.currentDate()
        current_date = now.toString('dd MMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
        self.label_date.setText(current_date)
        self.label_time.setText(current_time)

        # Manipulate QLabel Datetime Thread
        self.datetime_thread = DatetimeThread()
        self.datetime_thread.update_time.connect(self.UpdateTime)
        GenerateReport.generate_para_report("OpenCV Thread start")
        self.datetime_thread.start()

        # Manipulate Qlabel Video Camera Thread
        self.VideoThread = VideoThread()
        self.VideoThread.start()
        self.VideoThread.ImageUpdate.connect(self.ImageUpdateSlot)

        # Manipulate Qlabel Information Log
        self.label_info.setText("System Booting")
        self.label_info.setStyleSheet("color: red")

        # Manipulate QButton Change Room ID
        # TODO Feature Change Room + Implementation in attendance

        # Manipulate QButton Change Report Generation
        # TODO Feature Report Attendance

        # Manipulate QButton to Save Read New Face
        # TODO Feature Save New Face


        # Interact QButton Exit System
        self.btn_exit.clicked.connect(self.ExitSystem)

    def UpdateTime(self, this_date, this_time):
        self.label_date.setText(this_date)
        self.label_time.setText(this_time)

    def ImageUpdateSlot(self, image):
        self.label_camera.setPixmap(QPixmap.fromImage(image))
        # self.btn_settings.setDisabled(False)
        # self.btn_report.setDisabled(False)
        # self.btn_save_new.setDisabled(False)
        self.btn_exit.setDisabled(False)
        if not self.label_info.text() == "System Online":
            self.label_info.setText("System Online")
            self.label_info.setStyleSheet("color: green")

    def ExitSystem(self):
        self.label_info.setText("System Terminated")
        self.label_info.setStyleSheet("color: red")
        self.VideoThread.stop()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_date.setText(_translate("MainWindow", "-"))
        self.label_2.setText(_translate("MainWindow", "Time :"))
        self.label_time.setText(_translate("MainWindow", "-"))
        self.label.setText(_translate("MainWindow", "Date :"))
        self.label_3.setText(_translate("MainWindow", "Room ID"))
        self.label_room_id.setText(_translate("MainWindow", room_id))
        self.label_info.setText(_translate("MainWindow", "TextLabel"))
        self.btn_settings.setText(_translate("MainWindow", "Settings"))
        self.btn_report.setText(_translate("MainWindow", "Generate Report"))
        self.btn_save_new.setText(_translate("MainWindow", "Register Face"))

        self.btn_exit.setText(_translate("MainWindow", "End System"))


class DatetimeThread(QThread):
    update_time = pyqtSignal(str, str)

    def run(self):
        current_date = QDateTime.currentDateTime().toString("dd MMM yyyy")
        current_time = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.update_time.emit(current_date, current_time)


class VideoThread(QThread):
    ImageUpdate = pyqtSignal(QImage)

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
        GenerateReport.generate_para_report("Video Init 1280x720")
        cap, height, width, writer = video_init(camera_source=self.camera_source,
                                                resolution="720",
                                                to_write=False,
                                                save_dir=None)

        # Face detection init (GPU RATION 0.1)
        GenerateReport.generate_para_report("Face Detection Load 10% GPU if available")
        fmd = FaceMaskDetection(self.face_mask_model_path, self.margin, GPU_ratio=None)

        # Face recognition init
        GenerateReport.generate_para_report("Face Recognition with GPU if available")
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

            # Read images from the database
            GenerateReport.generate_para_report("Read Images and store in embeddings")
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
                GenerateReport.generate_para_report(str("Embeddings finished within " + str(d_t) +
                                                        " seconds and gain " + str(len_ref_path) + "embeddings"))

            # Tensor setting for calculating distance
            if len_ref_path > 0:
                with tf.Graph().as_default():
                    tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
                    tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
                    tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
                    # GPU setting
                    config = tf.ConfigProto(log_device_placement=True,
                                            allow_soft_placement=True,
                                            )
                    config.gpu_options.allow_growth = True
                    sess_cal = tf.Session(config=config)
                    sess_cal.run(tf.global_variables_initializer())

                feed_dict_2 = {tf_ref: embeddings_ref}

        GenerateReport.generate_para_report("OpenCV Init [END]")
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

                cv2.putText(img, self.FPS, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                Image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(Image.data,
                                           Image.shape[1],
                                           Image.shape[0],
                                           QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1280, 720, Qt.KeepAspectRatio)
                # TODO Implement FPS Catch Report
                writefps = self.FPS
                GenerateReport.generate_fps_report(writefps)
                self.ImageUpdate.emit(Pic)

                if writer is not None:
                    writer.write(img)

        if writer is not None:
            writer.release()

    def stop(self):
        self.ThreadActive = False
        self.quit()


if __name__ == "__main__":
    import sys

    gettime = datetime.datetime.now().strftime("%H:%M:%S.%f")
    GenerateReport.generate_para_report(str("System Start Booting with GUI on " + gettime))
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
