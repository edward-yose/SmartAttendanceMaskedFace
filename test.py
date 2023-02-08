import tkinter as tk
import cv2
import threading

class AttendanceApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.geometry("400x400")
        self.title("Attendance System")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainingdata.yml")
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.label = tk.Label(self, text="", font=("Helvetica", 20))
        self.label.pack()
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()

    def video_loop(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 45 and conf <= 85:
                    self.label.config(text="Attendance Marked for Employee {}".format(id_))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
