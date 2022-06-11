from flask import Flask, render_template, Response, request
import os
import time
from imageProcessing import pipeline
import cv2
import numpy as np
import jsonpickle
import base64

app = Flask(__name__)

def format_server_time():
  server_time = time.localtime()
  return time.strftime("%I:%M:%S %p", server_time)

@app.route('/')
def index():
    context = { 'server_time': format_server_time() }
    return render_template('index.html', context=context)

@app.route('/api/predict', methods=["POST"])
def predict():
    print("predicting")
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print('image received. size={}x{}'.format(img.shape[1], img.shape[0]))

    res, image = pipeline(img)

    retval, buffer = cv2.imencode('.png', img)
    png_as_text = str(base64.b64encode(buffer)).split("'")[1]

    print(png_as_text[:80])
    response = {
        'prediction' : res,
        'image' : png_as_text
    }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # do some fancy processing here....

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))