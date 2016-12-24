import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import scipy.misc as sp

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
img_gen = None
shape = (80,160)
mask = np.zeros((80,160,3))
mask[0:27,:,:] = 0
mask[27:65,:,:] = 1
mask[65:,:,:] = 0
#mask = np.zeros((160,320,3))
#mask[0:55,:,:] = 0
#mask[55:130,:,:] = 1
#mask[130:,:,:] = 0


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    image_array = sp.imresize(image_array, size=shape, interp='cubic')
    image_array = np.multiply(image_array,mask).astype(np.uint8)
    #print(image_array.shape, image_array.sum())

    transformed_image_array = image_array[None, :, :, :]
    print(transformed_image_array.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    #steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    steering_angle = float(model.predict_generator(img_gen.flow(transformed_image_array, batch_size=1), val_samples=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    if (abs(steering_angle)<0.1):
        throttle = 0.2
    else:
        if (abs(steering_angle)<0.15):
            throttle = 0.05# * 1./(steering_angle if steering_angle>0.1)
        else:
    	    throttle = 0.01# * 1./(steering_angle if steering_angle>0.1)

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    img_gen = ImageDataGenerator()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
