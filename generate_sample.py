import cv2
from ultralytics import YOLO
import os

def main():
    model_path = "output/yolo_train_results/yolo11s_cards3/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolo11s.pt"

    model = YOLO(model_path)
    video_path = "data/VideoBlackjack.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Grab a frame from somewhere in the middle
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2)) # 2nd second
    
    ret, frame = cap.read()
    if ret:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imwrite("prediction_sample.jpg", annotated_frame)
        print("Sample saved to prediction_sample.jpg")
    
    cap.release()

if __name__ == "__main__":
    main()
