import cv2
from ultralytics import YOLO
import os
import sys

def main():
    # 1. Select the model
    # We look for the latest best.pt based on exploration
    model_path = "output/yolo_synthetic/train/weights/best.pt"
    if not os.path.exists(model_path):
        # Fallback to base model if custom one doesn't exist
        print(f"Custom model not found at {model_path}. Checking for base models...")
        if os.path.exists("yolo11s.pt"):
            model_path = "yolo11s.pt"
        elif os.path.exists("yolo11n.pt"):
            model_path = "yolo11n.pt"
        else:
            print("No suitable YOLO model found.")
            sys.exit(1)
            
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # 2. Open the video
    video_path = "../data/VideoBlackjack.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found at {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame Size: {width}x{height}")

    # We want 1 frame per second
    # If FPS is 30, we process frame 0, 30, 60...
    step = int(fps)
    if step == 0: step = 1

    current_frame_index = 0
    
    print("\nStarting prediction display Loop...")
    print("Press 'q' to quit, 'n' for next frame immediately.")

    while True:
        # Set the video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached.")
            break
            
        if current_frame_index == 0:
            cv2.imwrite("debug_first_frame.jpg", frame)
            print("Saved debug_first_frame.jpg")

        # Predict
        # verbose=False to keep clutter down
        # Lower confidence to catch weak detections, set imgsz to match training typically
        results = model(frame, verbose=False, conf=0.04)
        
        # Check if we have any detections
        num_detections = len(results[0].boxes)
        print(f"Frame {current_frame_index}: {num_detections} detections")

        # Plot the results on the frame
        annotated_frame = results[0].plot()

        # Display information on the frame
        timestamp = current_frame_index / fps
        cv2.putText(annotated_frame, f"Time: {timestamp:.1f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the image
        cv2.imshow("YOLO Prediction (1 fps)", annotated_frame)

        # Wait for 1000ms (1 second) so the user can see it, conforming to "1 frame per second" viewing?
        # Or maybe they just want to process samples. 2000ms gives time to inspect.
        key = cv2.waitKey(2000) 
        
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('n'):
            pass # just continue immediately

        # Move to next second
        current_frame_index += step
        
        if current_frame_index >= total_frames:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
