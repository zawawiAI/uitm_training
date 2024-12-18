import cv2
from ultralytics import YOLO
from tkinter import Tk, Canvas, Button, messagebox
from PIL import Image, ImageTk
import time
import logging
from threading import Thread, Event
import pyttsx3
import platform
import subprocess

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')

class ObjectDetectorApp:
    def __init__(self, master, model_path="wheel.pt", camera_index=1):
        self.master = master
        self.master.title("Wheel Detector")

        # Initialize text-to-speech engine with detailed logging
        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure TTS properties
            self.tts_engine.setProperty('rate', 150)  # Speaking rate
            self.tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        except Exception as e:
            logging.error(f"TTS Initialization Error: {e}")

        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.model.fuse()
        self.target_classes = ['wheel']

        # Set up GUI components
        self.canvas = Canvas(master, width=640, height=480)
        self.canvas.pack()

        self.detect_button = Button(master, text="Start Detection", command=self.start_continuous_detection)
        self.detect_button.pack()

        self.quit_button = Button(master, text="Quit", command=self.on_closing)
        self.quit_button.pack()

        # Camera initialization
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            logging.error("Camera not detected or cannot be opened.")
            messagebox.showerror("Camera Error", "No camera detected")

        # Detection parameters
        self.is_detecting = False
        self.detection_thread = None
        self.last_detection_time = 0

    def speak(self, message):
        """Text-to-speech method"""
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
            logging.info(f"Spoke: {message}")
        except Exception as e:
            logging.error(f"Speech error: {e}")

    def detect_loop(self):
        """Continuous detection loop"""
        while self.is_detecting:
            if not self.cap.isOpened():
                logging.error("Camera not available")
                break

            # Capture a single frame
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                continue

            # Check time since last detection (NOW SET TO 5 SECONDS)
            current_time = time.time()
            if current_time - self.last_detection_time < 5:
                time.sleep(1)
                continue

            # Run YOLO model inference
            results = self.model.predict(
                source=frame, 
                conf=0.25,
                verbose=False
            )

            # Process detections
            wheel_detected = False
            processed_frame = frame.copy()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    label_index = int(box.cls)
                    label = result.names[label_index]
                    confidence = box.conf[0].item()

                    # Check for wheel detection
                    if label == 'wheel' and confidence >= 0.25:
                        wheel_detected = True
                        logging.info(f"Wheel detected with confidence: {confidence:.2f}")

                        # Draw bounding box
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Wheel {confidence:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update GUI with processed frame
            img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
            self.canvas.imgtk = imgtk  # Keep reference

            # Text-to-speech and time tracking for wheel detection
            if wheel_detected:
                self.speak("Wheel detected")
                self.last_detection_time = current_time

            # Small delay to prevent overwhelming the system
            time.sleep(0.1)

    # Remaining methods stay the same as in previous version
    def start_continuous_detection(self):
        if not self.is_detecting:
            self.is_detecting = True
            self.detection_thread = Thread(target=self.detect_loop, daemon=True)
            self.detection_thread.start()
            self.detect_button.config(text="Stop Detection", command=self.stop_detection)

    def stop_detection(self):
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join()
        self.detect_button.config(text="Start Detection", command=self.start_continuous_detection)

    def on_closing(self):
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.master.destroy()

def main():
    root = Tk()
    app = ObjectDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
