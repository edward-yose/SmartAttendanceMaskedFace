import tkinter as tk
import cv2
import numpy as np
import threading


class VideoCapture:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.frame = None
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.capture.read()
            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Application")

        self.start_button = tk.Button(text="Start", command=self.start_webcam)
        self.start_button.pack()

        self.stop_button = tk.Button(text="Stop", command=self.stop_webcam, state='disabled')
        self.stop_button.pack()

        self.canvas = tk.Canvas(width=640, height=480)
        self.canvas.pack()

        self.capture = None
        self.thread = None
        self.stop_event = None
        self.photo = None

    def start_webcam(self):
        self.capture = VideoCapture().start()
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()

    def stop_webcam(self):
        self.capture.stop()
        self.thread.join()

        self.canvas.delete("frame")

        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def update_frame(self):
        while True:
            frame = self.capture.read()
            if frame is not None:
                self.photo = tk.PhotoImage(image=np.array(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="frame")
                self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
