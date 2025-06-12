import cv2
import datetime
import os

class VideoCaptureModule:
    def __init__(self, camera_id=0, save_dir="captures"):
        self.camera_id = camera_id
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не вдалося відкрити камеру з ID {self.camera_id}")

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Не вдалося зчитати кадр з камери")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.save_dir}/frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        return frame, filename

    def show_live(self):
        print("Натисніть 'q', щоб вийти з режиму перегляду")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            cv2.imshow("Live Feed", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def release(self):
        self.cap.release()