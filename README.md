# Drowsiness & Yawning Detection System 💤

A **real-time drowsiness and yawning detection system** built using **Python**, **OpenCV**, **dlib**, and **Tkinter**. This project monitors a person’s eye and mouth movements through the webcam and alerts the user when drowsiness or yawning is detected.

---

## Project Overview

This project detects:

- **Sleeping:** Eyes fully closed for a prolonged period.  
- **Drowsiness:** Eyes partially closed or blinking slowly.  
- **Yawning:** Mouth open beyond a threshold.  

It provides **real-time visual feedback** and **audio alerts**.

---

## Features

- Real-time face and landmark detection using `dlib`.  
- Eye aspect ratio (EAR) based blink and drowsiness detection.  
- Mouth aspect ratio (MAR) based yawning detection.  
- Visual feedback: Status messages displayed on video (`Active :)`, `Drowsy !`, `SLEEPING !!!`, `Yawning !!!`).  
- Audio alert: Beep sound triggers when prolonged eye closure is detected.  
- Tkinter GUI: Simple interface with a start button and background image.
  
---


## Requirements

- Python 3.12 
- OpenCV (`cv2`)  
- dlib  
- imutils  
- numpy  
- Pillow (`PIL`)  
- winsound (built-in on Windows)  
- Tkinter (built-in with Python)  

---

## File Structure

```text
drowsiness-detection/
├── drowsiness_detection.py                  # Main Python script
├── shape_predictor_68_face_landmarks.dat    # Dlib pretrained model
├── dark_gradient_img.jpeg                   # Background image for Tkinter GUI                      
├── README.md                                # Project documentation
└── requirements.txt                         # Python dependencies
```
---

Install dependencies using:

```bash
pip install opencv-python dlib imutils numpy pillow
