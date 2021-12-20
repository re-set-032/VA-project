import numpy as np
import sys
import math
import cv2
import io
import base64
from flask_restful import Resource, reqparse 
from flask import Flask, request, jsonify

import json
import matplotlib.pyplot as plt

# ngrok behaves as a mirror that forwards requests and reponses to and from the flask server
from quick_flask_server import run_with_ngrok
from PIL import Image  

mapping = {  # map that designates the bounding box location to any of the 9 regions
    0: "Top left",
    1: "Center Left",
    2: "Bottom Left",
    3: "Center Top",
    4: "Center",
    5: "Center Bottom",
    6: "Top Right",
    7: "Center Right",
    8: "Bottom Right"
}


def disect(h, w, x1, y1, h1, w1):
    # disection of image in vertical direction(3 regions, 1, 3, 5)
    div_ver = [x*h//6 for x in range(1, 6, 2)]
    # disection of image in horizontal direction(3 regions 1, 3, 5)
    div_hor = [x*w//6 for x in range(1, 6, 2)]
    count = 0
    mn = sys.maxsize
    index = 0  # mapping index
    for x in div_hor:
        for y in div_ver:
            if math.sqrt((x-x1-w1/2)**2 + (y-y1-h1/2)**2) < mn:
                mn = math.sqrt((x-x1-w1/2)**2 + (y-y1-h1/2)**2)
                index = count
            count += 1

    return mapping[index]


app = Flask(__name__)
run_with_ngrok(app)
yolo = cv2.dnn.readNet("trainedYolo.weights", "trainedYolo.cfg")
f = open("cocoClasses.names", "r")
objectClasses = [line.strip() for line in f.readlines()]


@app.route("/predict", methods=["POST", "GET"])
def predict():
    #meta = json.load(request.files['meta'])
    data = request.get_json()

    if 'image' not in data:
        return "", 400

    img = base64.b64decode(data['image'])

    img = Image.open(io.BytesIO(img))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite('rep.jpeg', img)

    layers = yolo.getLayerNames()
    outputLayerIndices = yolo.getUnconnectedOutLayers()
    outputLayers = [layers[i[0] - 1] for i in outputLayerIndices]

    img = cv2.resize(img, None, fx=0.4, fy=0.3)
    height, width, channels = img.shape
    blobs = cv2.dnn.blobFromImage(
        img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blobs)
    outputs = yolo.forward(outputLayers)

    object_ids = []
    possibilities = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            object_id = np.argmax(scores)
            possibility = scores[object_id]

            if(possibility > 0.5):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                object_ids.append(object_id)
                possibilities.append(float(possibility))
                boxes.append([x, y, w, h])

    # removing the multiple similar objects
    uniqueIndices = cv2.dnn.NMSBoxes(boxes, possibilities, 0.4, 0.6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    object_list = []
    h, w = img.shape[0:2]
    output = ""

    for i in range(len(boxes)):
        if i in uniqueIndices:
            x, y, w1, h1 = boxes[i]
            label = str(objectClasses[object_ids[i]])
            object_list.append([x, y, w1, h1, label])
            loc = disect(h, w, x, y, h1, w1)
            output += f'{label} is in {loc}                     '

    print(object_list)

    if not object_list:
        return "nothing found"
    else:
        return output


if __name__ == '__main__':
    app.run()
