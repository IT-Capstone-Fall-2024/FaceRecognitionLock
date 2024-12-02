import threading
import cv2
import time
import paho.mqtt.client as mqtt
from deepface import DeepFace

# Initialize video capture and mqtt
cap = cv2.VideoCapture(1)
start = time.time()
broker = "192.168.90.234"
port = 1883
client = mqtt.Client(client_id="", userdata=None, protocol=mqtt.MQTTv5)
client.connect(broker, port)
topic = "lock/1"

# Set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load multiple reference images
reference_images = [
    cv2.imread("kyle.png"),
    cv2.imread("Andrew.png")
]

# Function to check face match
def check_face(frame):
    global face_match
    try:
        # Check against each reference image
        for ref_img in reference_images:
            if DeepFace.verify(frame, ref_img.copy())['verified']:
                face_match = True
                return  # Exit as soon as a match is found
        face_match = False  # No match found after checking all reference images
    except ValueError:
        face_match = False

# Main video processing loop
while True:
    ret, frame = cap.read()
    end = time.time()
    countdown = (end - start)
    if ret:
        if counter % 60 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1

        # Display match status
        if face_match:
            client.publish(topic, "unlock", qos=1)
            break
        elif countdown >= 20:
            client.publish(topic, "lock", qos=1)
            break

        cv2.imshow("video", frame)

    # Exit if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
