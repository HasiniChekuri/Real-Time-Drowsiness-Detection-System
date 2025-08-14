import cv2
import numpy as np
import dlib
from imutils import face_utils
import winsound
import threading
import tkinter as tk
from PIL import Image, ImageTk

# ------------------------
# Detector setup
# ------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r"C:\Users\hasin\Documents\Drowsiness Detection\shape_predictor_68_face_landmarks.dat"
)

# ------------------------
# Status variables
# ------------------------
sleep = 0
drowsy = 0
active = 0
yawn = 0
status = ""
color = (0, 0, 0)
sound_thread_running = False

# ------------------------
# Helper functions
# ------------------------
def play_alert_sound():
    """Play alert sound in a separate thread."""
    global sound_thread_running
    if not sound_thread_running:
        sound_thread_running = True
        try:
            winsound.Beep(1000, 2000)
        finally:
            sound_thread_running = False

def compute(ptA, ptB):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    """Detect eye state: open, partially closed, or sleeping."""
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2   # Open
    elif ratio > 0.21:
        return 1   # Partial Blink
    else:
        return 0   # Sleeping

def is_yawning(landmarks):
    """Detect yawning based on mouth aspect ratio."""
    top_lip = landmarks[51]
    bottom_lip = landmarks[57]
    left_corner = landmarks[48]
    right_corner = landmarks[54]

    vertical_dist = compute(top_lip, bottom_lip)
    horizontal_dist = compute(left_corner, right_corner)
    yawn_ratio = vertical_dist / horizontal_dist

    return yawn_ratio > 0.6

# ------------------------
# Detection function
# ------------------------
def start_detection():
    global sleep, drowsy, active, yawn, status, color

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # HD width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # HD height

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_frame = frame.copy()

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                                 landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                                  landmarks[47], landmarks[46], landmarks[45])

            if is_yawning(landmarks):
                yawn += 1
                if yawn > 5:
                    status = "Yawning !!!"
                    color = (255, 165, 0)
            else:
                yawn = 0

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = active = 0
                if sleep > 10:
                    status = "SLEEPING !!!"
                    color = (0, 0, 255)
                    threading.Thread(target=play_alert_sound).start()
            elif left_blink == 1 or right_blink == 1:
                sleep = active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 255, 255)
            else:
                sleep = drowsy = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

            cv2.putText(face_frame, status, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("Drowsiness Detection", face_frame)

        # Exit if OpenCV window closed or Esc pressed
        if cv2.getWindowProperty("Drowsiness Detection", cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            return
        if cv2.waitKey(1) & 0xFF == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

# ------------------------
# Tkinter UI
# ------------------------
root = tk.Tk()
root.title("Drowsiness Detection")
root.geometry("800x600")

# Load background image
bg_image = Image.open(r"C:\Users\hasin\Documents\Drowsiness Detection\dark gradient img.jpeg")
bg_image = bg_image.resize((1280, 720), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Start button
start_button = tk.Button(root, text="Start Detection",
                         font=("Arial", 20, "bold"),
                         bg="green", fg="white",
                         command=start_detection)
start_button.place(relx=0.5, rely=0.5, anchor="center")

root.mainloop()
