from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import os

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def draw_bboxes(self, frames, ball_detections):
        output_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Ball", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            output_frames.append(frame)
        return output_frames

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# === Run Inference ===
video_path = "Test1.mp4"
trained_model_path = "deakin\Deakin Year 3\Sem1\Group projects\Model attempt 1 - NFL\runs\detect\nfl-ball-detector3\weights\best.pt"

tracker = BallTracker(trained_model_path)

frames = load_video_frames(video_path)
detections = tracker.detect_frames(frames)
output_frames = tracker.draw_bboxes(frames, detections)
save_video(output_frames, "nfl_ball_output.mp4")
