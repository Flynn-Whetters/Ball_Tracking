from ultralytics import YOLO
import cv2

model_path = r"D:\deakin\Deakin Year 3\Sem1\Group projects\Model attempt 1 - NFL\runs\detect\nfl-ball-detector3\weights\best.pt"  # Adjust if needed
try:
    model = YOLO(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Open video file
video_path = "D:/deakin/Deakin Year 3/Sem1/Group projects/Model attempt 1 - NFL/Test1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving

# Save output video
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLO on the frame
    results = model.predict(frame, conf=0.2)  # Adjust confidence threshold as needed

    # Draw bounding boxes
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)  # Get bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw box

    # Write to output file
    out.write(frame)

    # Show video (optional)
    cv2.imshow("NFL Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()