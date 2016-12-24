import argparse
import base64
import json

import numpy as np
import socketio
from socketio import Namespace

import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template

from io import BytesIO

class ControlServer(Namespace):

    def __init__(self):
        super().__init__()
        self.sio = None
        self.callbacks = []


    def start(self):
        print('inside control_srv.start()')
        self.sio = socketio.Server()
        print('sio')
        self.sio.register_namespace(self)
        print('register')
        self.app = socketio.Middleware(self.sio, Flask(__name__))
        print('middleware')
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)
        print('wsgi server')

    def register_callback(self, cb):
        self.callbacks.append(cb)
        def unsubscribe():
            self.callbacks.remove(cb)
        return unsubscribe

    def on_telemetry(self, sid, data):
        #print('server.py on_telemetery',data)
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))

        telemetry = {'steering_angle': steering_angle,
                     'throttle': throttle,
                     'speed': speed,
                     'image': np.asarray(image, dtype=np.float32)}

        for cb in self.callbacks:
            cb.handle_telemetry(telemetry)

    def on_connect(self, sid, environ):
        print("connect ", sid)
        for cb in self.callbacks:
            cb.handle_connect(sid)
        self.send_control(0, 0)

    def send_control(self, steering_angle, throttle):
        self.sio.emit("steer", data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        }, skip_sid=True)
