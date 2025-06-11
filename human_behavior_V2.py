import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque

# Constants
FIGHT_WINDOW = 10
fight_buffer = deque(maxlen=FIGHT_WINDOW)
YOLO_POSE_MODEL_PATH = "YOLO/yolov8s-pose.pt"
CONF_THRESHOLD = 0.5

# Utility Functions
def calculate_keypoint_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def analyze_pose_for_fight(kp1, kp2, frame_width, frame_height):
    if kp1 is None or kp2 is None:
        return False

    indicators = 0

    def visible(kp): return kp[2] > CONF_THRESHOLD

    def close(p1, p2, ratio):
        return calculate_keypoint_distance(p1, p2) < frame_width * ratio

    #kp1 and kp2 is person1 and person2

    # Right wrist to nose
    if visible(kp1[10]) and visible(kp2[0]) and close(kp1[10], kp2[0], 0.07):
        indicators += 1
    #kp1[10] = werists(Right), kp1[9] = werists(Left)
    # Left wrist to neck
    if visible(kp1[9]) and visible(kp2[5]) and visible(kp2[6]):
        neck = [(kp2[5][0] + kp2[6][0]) / 2, (kp2[5][1] + kp2[6][1]) / 2]
        if close(kp1[9], neck, 0.07): indicators += 1
    # kp1[0] = Nose
    # kp1[5][6] = shoulders
    # Right wrist to nose (person 2 to person 1)
    if visible(kp2[10]) and visible(kp1[0]) and close(kp2[10], kp1[0], 0.07):
        indicators += 1
    # Left wrist to neck (person 2 to person 1)
    if visible(kp2[9]) and visible(kp1[5]) and visible(kp1[6]):
        neck = [(kp1[5][0] + kp1[6][0]) / 2, (kp1[5][1] + kp1[6][1]) / 2]
        if close(kp2[9], neck, 0.07): indicators += 1
    # hip proximity
    if all(visible(kp1[i]) for i in [11, 12]) and all(visible(kp2[i]) for i in [11, 12]):
        mid1 = [(kp1[11][0] + kp1[12][0]) / 2, (kp1[11][1] + kp1[12][1]) / 2]
        mid2 = [(kp2[11][0] + kp2[12][0]) / 2, (kp2[11][1] + kp2[12][1]) / 2]
        if close(mid1, mid2, 0.15): indicators += 1
    # kp[11][12] = Hips(Left, Right)

    return indicators >= 3

def detect_fight_in_frame(frame, model):
    H, W = frame.shape[:2]
    results = model(frame, verbose=False)[0]
    people = []

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if model.names[cls] != "person" or conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        kps = results.keypoints.data[i].cpu().numpy().tolist() if results.keypoints is not None else None

        people.append({'box': (x1, y1, x2, y2), 'keypoints': kps})

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if kps:
            for kp in kps:
                if kp[2] > CONF_THRESHOLD:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 0, 255), -1)

    fight_detected = False
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1, p2 = people[i], people[j]
            center1 = ((p1['box'][0] + p1['box'][2]) // 2, (p1['box'][1] + p1['box'][3]) // 2)
            center2 = ((p2['box'][0] + p2['box'][2]) // 2, (p2['box'][1] + p2['box'][3]) // 2)
            dist = calculate_keypoint_distance(center1, center2)

            if dist < W * 0.3 and analyze_pose_for_fight(p1['keypoints'], p2['keypoints'], W, H):
                fight_detected = True
                for box in [p1['box'], p2['box']]:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Fight Detected",
                            (min(p1['box'][0], p2['box'][0]), min(p1['box'][1], p2['box'][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, fight_detected

def process_video(video_path):
    model = YOLO(YOLO_POSE_MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, fight_detected = detect_fight_in_frame(frame, model)
        fight_buffer.append(fight_detected)

        if sum(fight_buffer) > FIGHT_WINDOW * 0.6:
            cv2.putText(processed_frame, "Stable Fight Alert", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        cv2.imshow("Fight Detection", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("p.mov")
