import requests
import cv2
import base64

# address to API 
addr = 'ADDRESS TO SERVER'

# API endpoint
endpoint = '/api/predict'

# destination
url = addr + endpoint

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# load the image
imageIn = "./test.jpeg"
img = cv2.imread(imageIn)

# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)

# send http request with image and receive response
response = requests.post(url, data=img_encoded.tobytes(), headers=headers)

# output the prediction
print(response.json()['prediction'])

# convert decode base64
base64_img = response.json()['image']
bytes_img = bytes(base64_img, 'utf-8')
binary_img = base64.b64decode(bytes_img)

# save image response
with open("prediction.png", "wb") as fh:
   fh.write(binary_img)

# show the output image
cv2.imshow("Response",cv2.imread("prediction.png"))
cv2.waitKey(0)

