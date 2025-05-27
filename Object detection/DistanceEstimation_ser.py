import cv2 as cv
import numpy as np
import serial
import time

# Distance constants 
KNOWN_DISTANCE = 35  # inches
PERSON_WIDTH = 16
MOBILE_WIDTH = 3.0

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors and fonts
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONTS = cv.FONT_HERSHEY_COMPLEX

# Serial setup
dev_ser = "COM3" 
ser = serial.Serial(dev_ser, 9600, timeout=2)

# Cooldown control
last_sent = {}
serial_delay = 3  # seconds

# Load class names
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Serial command map
serial_command_map = {
    "person": "1",
    "cell phone": "2",
    "bottle": "3",
    "chair": "4"
}

# Load YOLOv3 model
yoloNet = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Serial sender with delay
def send_serial(obj_type):
    current_time = time.time()
    if obj_type in serial_command_map:
        if obj_type not in last_sent or (current_time - last_sent[obj_type]) > serial_delay:
            ser.write((serial_command_map[obj_type] + '\n').encode())
            print(ser.write((serial_command_map[obj_type] + '\n').encode()))
            last_sent[obj_type] = current_time

# Distance finder functions
def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width

def distance_finder(focal_length, real_object_width, width_in_frame):
    return (real_object_width * focal_length) / width_in_frame

# Object detection function
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        class_id = classid[0]
        label = class_names[class_id]
        color = COLORS[class_id % len(COLORS)]
        cv.rectangle(image, box, color, 2)
        cv.putText(image, f"{label}: {score.item():.2f}", (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        if label in serial_command_map.keys():
            data_list.append([label, box[2], (box[0], box[1] - 2)])
    return data_list

# Reference images to calibrate focal length
ref_person = cv.imread('ReferenceImages/image3.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

# Compute focal lengths
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

# Start camera
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = object_detector(frame)
    for d in data:
        name, width, (x, y) = d
        if name == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, width) 
        elif name == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, width)
        else:
            distance = None

        send_serial(name)

        if distance:
            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv.imshow('Object Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
