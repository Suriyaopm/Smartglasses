import pyttsx3
import cv2
from ultralytics import YOLO
import concurrent.futures
import time
import numpy as np

# Load YOLOv8m model
model = YOLO("yolov8m.pt")

# Assign unique colors to different classes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Add more colors as needed

def speak(text, female_voice=True):
    engine = pyttsx3.init()
    engine.setProperty('rate', 230)  # Speed of speech
    engine.setProperty('volume', 2)
    voices = engine.getProperty('voices')
    print(text)
    if female_voice:
        engine.setProperty('voice', voices[1].id)  # Setting female voice
    else:
        engine.setProperty('voice', voices[0].id)  # Setting male voice
    engine.say(text)
    engine.runAndWait()

# Function to determine position of object
def get_position(frame_width, box_x_min, box_x_max):
    frame_center = frame_width // 2
    box_center = (box_x_min + box_x_max) // 2
    if box_center < frame_center - 50:
        return "left"
    elif box_center > frame_center + 50:
        return "right"
    else:
        return "center"

# Function to estimate distance in feet (rough estimation)
def estimate_distance(pixel_width):
    # Assuming known object width in feet and its corresponding pixel width
    object_width_feet = 1.0  # Change this value according to your scenario
    object_pixel_width = 100  # Change this value according to your scenario

    # Calculate distance based on simple linear scaling
    distance_feet = object_width_feet * object_pixel_width / pixel_width
    return distance_feet

# Function for real-time object detection
def detect_objects(frame):
    # Perform object detection on the frame
    results = model.predict(frame)

    # Get the first result
    result = results[0]

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Initialize dictionaries to store counts of objects in different positions
    objects_left = {}
    objects_right = {}
    objects_center = {}

    wall_detected = False

    for i, box in enumerate(result.boxes):
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        prob = box.conf[0].item()

        # Determine position of object
        position = get_position(frame_width, cords[0], cords[2])


        # Store object count in the respective position dictionary
        if position == "left":
            if label in objects_left:
                objects_left[label] = max(objects_left[label], estimate_distance(cords[2] - cords[0]))
            else:
                objects_left[label] = estimate_distance(cords[2] - cords[0])
        elif position == "right":
            if label in objects_right:
                objects_right[label] = max(objects_right[label], estimate_distance(cords[2] - cords[0]))
            else:
                objects_right[label] = estimate_distance(cords[2] - cords[0])
        else:
            if label in objects_center:
                objects_center[label] = max(objects_center[label], estimate_distance(cords[2] - cords[0]))
            else:
                objects_center[label] = estimate_distance(cords[2] - cords[0])

        # Draw bounding box
        color = colors[i % len(colors)]
        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)
        cv2.putText(frame, f'{label}: {prob:.2f}', (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Check if the frame is fully black
    if np.sum(frame) == 0:
        speak("I can't see anything")
    elif (np.sum(frame) <= 3):
        if label == "person":
            if not objects_right:  # If no objects on the right
                speak(f"{label} in front of you, turn right to avoid!")# If no objects on the right
            if not objects_left:
                speak(f"{label} in front of you, turn left to avoid!")


    elif np.sum(frame) <= 6:
        speak("More objects are predicted in the frame Turn right or left to avoid")
    else:
        # Speak the names of objects detected in the center position
        if objects_center:
            for label, distance in objects_center.items():
                if distance <= 9:  # Object within 7 feet
                    if not objects_right:  # If no objects on the right
                        speak(f"{label} in front of you, turn right to avoid!")
                    if not objects_left:
                        speak(f"{label} in front of you, turn left to avoid!")
        else:
            speak("No object is predicted in front of you")
            if objects_center:
                for label, distance in objects_center.items():
                    if distance <= 9:  # Object within 7 feet
                        if not objects_right:  # If no objects on the right
                            speak(f"{label} in front of you, turn right to avoid!")
                        if not objects_left:
                            speak(f"{label} in front of you, turn left to avoid!")

    return frame
#
def start_video_capture():
    #url = "http://192.168.190.117:81/stream"
    url = "http://192.168.232.117:81/stream"
    # url = 0
    cap = cv2.VideoCapture(url)
    time.sleep(1.0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame. Restarting capture...")
            cap.release()
            time.sleep(1.0)
            start_video_capture()


        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(detect_objects, frame)
            frame_with_boxes = future.result()

        cv2.imshow('Object Detection', frame_with_boxes)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # time.sleep(0.05/7)
        time.sleep(3)
    cap.release()
    cv2.destroyAllWindows()

if _name_ == "_main_":
    start_video_capture()
