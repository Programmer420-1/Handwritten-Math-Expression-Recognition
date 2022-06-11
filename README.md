# Handwritten-Math-Expression-Recognition
Simple Flask API that accepts image of handwritten math expression and responds with extracted characters from the image.

## Background
This project is built for the assignment of WIE2003 Introduction to Data Science which is taken in Universiti Malaya as R's opencv module's functionality is not enough to solve the problems we faced in this assignment

## Server Setup
1. Install necessary dependencies via ```pip install -r requirement.txt```.
2. Navigate to the directory where ```app.py``` is at.
3. Run ```app.py```.
4. A Flask server should be running at ```http://localhost:8080``` or the local network IP of your devive at port 8080.

## Client Setup
1. Change the value of ```addr``` in ```client.py``` to the IP of the server.
```python
  addr = "http://127.0.0.1:8080"
```
2. Save and run the file, the prediction string will be shown in the terminal together with the confidence value of each character. An image with bounding box drawn will also be displayed.

## Credits
Acquired the datasets from [here](www.google.com)
