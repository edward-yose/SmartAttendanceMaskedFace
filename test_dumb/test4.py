import tkinter as tk
import cv2
import threading
import PIL.Image, PIL.ImageTk


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

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.update_frame, args=(self.stop_event,))
        self.thread.start()

    def stop_webcam(self):
        self.stop_event.set()
        self.thread.join()

        self.capture.release()
        self.canvas.delete("frame")

        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def update_frame(self, stop_event):
        while not stop_event.is_set():
            ret, frame = self.capture.read()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor="nw", tags="frame")
                self.root.update()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()
