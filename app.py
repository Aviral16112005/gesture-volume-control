from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np

from ctypes import cast, POINTER
import comtypes
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

app = Flask(__name__)

running = False
current_volume = 0
finger_distance = 0

# ================= AUDIO SETUP =================
devices = AudioUtilities.GetSpeakers()

# FIX: proper activation
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

# Get volume range
vol_range = volume_interface.GetVolumeRange()
minVol = vol_range[0]
maxVol = vol_range[1]

# ================= HAND TRACKING =================
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
draw = mp.solutions.drawing_utils

# ================= VIDEO GENERATOR =================
def generate_frames():
    global running, current_volume, finger_distance

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if running and result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

                lm = hand.landmark

                x1, y1 = int(lm[4].x * w), int(lm[4].y * h)  # thumb
                x2, y2 = int(lm[8].x * w), int(lm[8].y * h)  # index

                # draw points
                cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 8, (0, 255, 255), -1)

                # line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # distance
                dist = math.hypot(x2 - x1, y2 - y1)
                finger_distance = int(dist)

                # volume mapping
                current_volume = int(np.interp(dist, [20, 200], [0, 100]))
                vol = np.interp(current_volume, [0, 100], [minVol, maxVol])

                volume_interface.SetMasterVolumeLevel(vol, None)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ================= ROUTES =================

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/start')
def start():
    global running
    running = True
    return jsonify({"status": "started"})


@app.route('/stop')
def stop():
    global running
    running = False
    return jsonify({"status": "stopped"})


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/volume')
def volume():
    return jsonify({
        "volume": current_volume,
        "distance": finger_distance
    })


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)