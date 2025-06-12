import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import threading
from event_logger import EventLogger
from face_database import FaceDatabase
from yolo_face_detector import YOLOFaceDetector
from facenet_embedder import FaceEmbedder
from face_identifier import FaceIdentifier

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система розпізнавання облич")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        self.root.configure(bg="#FBCEB1")
        self.logger = EventLogger()

        self.label = tk.Label(self.root, text="Вітаємо у системі", font=("Helvetica", 16), bg="#FBCEB1")
        self.label.pack(pady=20)

        self.btn_identify = tk.Button(self.root, text="Ідентифікація користувача", width=30, height=2, command=self.open_identification, bg="#FF9966")
        self.btn_identify.pack(pady=10)

        self.btn_admin = tk.Button(self.root, text="Вхід для адміністратора", width=30, height=2, command=self.open_admin_login, bg="#FF9966")
        self.btn_admin.pack(pady=10)

    def open_identification(self):
        self.root.withdraw()
        self.new_window = tk.Toplevel(self.root)
        self.new_window.title("Розпізнавання користувача")
        self.new_window.geometry("700x500")
        self.new_window.configure(bg="#FBCEB1")

        self.video_label = tk.Label(self.new_window)
        self.video_label.pack()

        self.stop_event = threading.Event()
        self.cap = cv2.VideoCapture(0)

        self.detector = YOLOFaceDetector(model_path="best.pt")
        self.embedder = FaceEmbedder()
        self.identifier = FaceIdentifier(threshold=0.6)

        self.update_frame()

        def on_close():
            self.close_video()
            self.root.deiconify()

        self.new_window.protocol("WM_DELETE_WINDOW", on_close)

    def update_frame(self):
        if self.stop_event.is_set():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        detections = self.detector.detect_faces(frame)
        names = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det[:4])
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                names.append(("Невідомий", 0.0))
                continue

            try:
                embedding = self.embedder.get_embedding(face_crop)
                name, score = self.identifier.identify(embedding)
                names.append((name, score))
                self.logger.log(name, score, "success")
            except Exception as e:
                print(f"Помилка при обробці обличчя: {e}")
                names.append(("Помилка", 0.0))
                self.logger.log("Помилка", 0.0, "fail")

        for (det, (name, score)) in zip(detections, names):
            x1, y1, x2, y2 = map(int, det[:4])
            label = f"{name} ({score:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.new_window.after(10, self.update_frame)

    def close_video(self):
        self.stop_event.set()
        self.cap.release()
        self.identifier.close()
        self.new_window.destroy()

    def open_admin_login(self):
        self.root.withdraw()
        self.login_window = tk.Toplevel(self.root)
        self.login_window.title("Вхід адміністратора")
        self.login_window.configure(bg="#FBCEB1")
        self.login_window.geometry("300x200")

        tk.Label(self.login_window, text="Логін:").pack(pady=5)
        self.entry_username = tk.Entry(self.login_window)
        self.entry_username.pack()

        tk.Label(self.login_window, text="Пароль:").pack(pady=5)
        self.entry_password = tk.Entry(self.login_window, show="*")
        self.entry_password.pack()

        tk.Button(self.login_window, text="Увійти", bg="#FF9966", command=self.verify_admin).pack(pady=15)

        def on_close():
            self.login_window.destroy()
            self.root.deiconify()

        self.login_window.protocol("WM_DELETE_WINDOW", on_close)

    def verify_admin(self):
        username = self.entry_username.get()
        password = self.entry_password.get()

        if username == "admin" and password == "1234":
            self.logger.log("admin", 1.0, "login_success", source="admin_panel")
            self.login_window.destroy()
            self.open_admin_panel()
        else:
            self.logger.log("admin", 0.0, "login_failed", source="admin_panel")
            messagebox.showerror("Помилка", "Неправильний логін або пароль")

    def open_admin_panel(self):
        self.admin_window = tk.Toplevel(self.root)
        self.admin_window.title("Панель адміністратора")
        
        self.admin_window.geometry("400x350")
        self.admin_window.configure(bg="#FBCEB1")

        tk.Label(self.admin_window, text="Панель адміністратора", font=("Helvetica", 14)).pack(pady=10)
        tk.Button(self.admin_window, text="Зареєструвати з вебкамери", width=30, command=self.register_user, bg="#FF9966").pack(pady=5)
        tk.Button(self.admin_window, text="Зареєструвати з фотографії", width=30, command=self.register_from_image, bg="#FF9966").pack(pady=5)
        tk.Button(self.admin_window, text="Ідентифікація з фотографії", width=30, command=self.identify_from_image, bg="#FF9966").pack(pady=5)
        tk.Button(self.admin_window, text="Переглянути лог подій", width=30, command=self.view_logs, bg="#FF9966").pack(pady=5)
        tk.Button(self.admin_window, text="Закрити", width=30, bg="#FF9966", command=lambda: [self.admin_window.destroy(), self.root.deiconify()]).pack(pady=10)

    def register_user(self):
        self.admin_window.withdraw()
        name = simpledialog.askstring("Реєстрація", "Введіть ім’я нового користувача:")
        if not name:
            self.admin_window.deiconify()
            return

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Помилка", "Не вдалося захопити зображення з камери")
            self.admin_window.deiconify()
            return

        detector = YOLOFaceDetector(model_path="best.pt")
        faces = detector.detect_faces(frame)
        if len(faces) == 0:
            messagebox.showwarning("Увага", "Обличчя не знайдено")
            self.admin_window.deiconify()
            return

        x1, y1, x2, y2 = map(int, faces[0][:4])
        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (160, 160))

        confirm_window = tk.Toplevel(self.admin_window)
        confirm_window.title("Підтвердження зображення")
        confirm_window.geometry("200x240")
        confirm_window.configure(bg="#FBCEB1")

        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_face)
        imgtk = ImageTk.PhotoImage(image=img)

        tk.Label(confirm_window, text="Це ви?").pack(pady=5)
        panel = tk.Label(confirm_window, image=imgtk)
        panel.image = imgtk
        panel.pack()

        def on_confirm():
            embedder = FaceEmbedder()
            embedding = embedder.get_embedding(face_crop)

            db = FaceDatabase()
            db.add_embedding(name, embedding)
            db.close()

            folder = "captures/registered"
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{name}.jpg")
            cv2.imwrite(path, face_crop)

            messagebox.showinfo("Готово", f"Користувач '{name}' зареєстрований успішно")
            confirm_window.destroy()
            self.admin_window.deiconify()

        def on_cancel():
            confirm_window.destroy()
            self.admin_window.deiconify()

        tk.Button(confirm_window, text="Підтвердити", command=on_confirm).pack(pady=5)
        tk.Button(confirm_window, text="Скасувати", command=on_cancel).pack(pady=5)

    def register_from_image(self):
        image_path = filedialog.askopenfilename(title="Оберіть фотографію", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not image_path:
            return

        frame = cv2.imread(image_path)
        detector = YOLOFaceDetector(model_path="best.pt")
        faces = detector.detect_faces(frame)

        if len(faces) == 0:
            messagebox.showerror("Помилка", "Обличчя не знайдено на зображенні.")
            return

        x1, y1, x2, y2 = map(int, faces[0][:4])
        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (160, 160))

        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_face)
        imgtk = ImageTk.PhotoImage(image=img)

        confirm_window = tk.Toplevel(self.admin_window)
        confirm_window.title("Підтвердження обличчя")
        confirm_window.configure(bg="#FBCEB1")
        confirm_window.geometry("200x350")

        tk.Label(confirm_window, text="Підтвердіть особу").pack(pady=5)
        label = tk.Label(confirm_window, image=imgtk)
        label.image = imgtk
        label.pack()

        name_entry = tk.Entry(confirm_window)
        name_entry.pack(pady=5)
        name_entry.insert(0, "Ім’я користувача")

        def on_confirm():
            name = name_entry.get()
            if not name:
                return
            embedder = FaceEmbedder()
            embedding = embedder.get_embedding(face_crop)

            db = FaceDatabase()
            db.add_embedding(name, embedding)
            db.close()

            os.makedirs("captures/registered", exist_ok=True)
            path = os.path.join("captures/registered", f"{name}.jpg")
            cv2.imwrite(path, face_crop)

            self.logger.log(name, 1.0, "register_success", source="admin_image", image_path=path)
            messagebox.showinfo("Готово", f"Користувача '{name}' додано")
            confirm_window.destroy()

        def on_cancel():
            confirm_window.destroy()

        tk.Button(confirm_window, text="Підтвердити", command=on_confirm).pack(pady=5)
        tk.Button(confirm_window, text="Скасувати", command=on_cancel).pack(pady=5)

    def identify_from_image(self):
        image_path = filedialog.askopenfilename(title="Оберіть фотографію", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not image_path:
            return

        frame = cv2.imread(image_path)
        detector = YOLOFaceDetector(model_path="best.pt")
        faces = detector.detect_faces(frame)

        if len(faces) == 0:
            messagebox.showerror("Помилка", "Обличчя не знайдено на зображенні.")
            return

        x1, y1, x2, y2 = map(int, faces[0][:4])
        face_crop = frame[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (160, 160))

        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_face)
        imgtk = ImageTk.PhotoImage(image=img)

        preview = tk.Toplevel(self.admin_window)
        preview.title("Результат ідентифікації")
        preview.configure(bg="#FBCEB1")
        preview.geometry("240x300")

        tk.Label(preview, text="Попередній перегляд").pack(pady=5)
        label = tk.Label(preview, image=imgtk)
        label.image = imgtk
        label.pack()

        identifier = FaceIdentifier()
        name, score = identifier.identify(FaceEmbedder().get_embedding(face_crop))
        identifier.close()

        self.logger.log(name, score, "identify_from_image", source="admin_image", image_path=image_path)
        result_text = f"Ім’я: {name}\nТочність: {score:.2f}"
        tk.Label(preview, text=result_text).pack(pady=10)
        tk.Button(preview, text="Закрити", command=preview.destroy).pack(pady=5)

    def view_logs(self):
        self.admin_window.withdraw()
        log_window = tk.Toplevel(self.admin_window)
        log_window.title("Журнал подій")
        log_window.geometry("800x400")
        log_window.configure(bg="#FBCEB1")

        text_area = scrolledtext.ScrolledText(log_window, wrap=tk.WORD, width=100, height=20)
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        cursor = self.logger.conn.cursor()
        cursor.execute("SELECT timestamp, name, score, status, source FROM access_log ORDER BY id DESC LIMIT 100;")
        records = cursor.fetchall()

        for row in records:
            timestamp, name, score, status, source = row
            line = f"[{timestamp}] {name} ({score:.2f}) → {status.upper()} через {source}\n"
            text_area.insert(tk.END, line)

        text_area.config(state=tk.DISABLED)

        def on_close():
            log_window.destroy()
            self.admin_window.deiconify()

        log_window.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
