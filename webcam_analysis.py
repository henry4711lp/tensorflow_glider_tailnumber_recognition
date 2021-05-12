import time
import urllib.request
import urllib3
import numpy as np
import keras_ocr
import cv2

from image_recognition import image_recognition


def url_to_image(url):
    url_response = urllib.request.urlopen(url)
    image = np.asarray(bytearray(url_response.read()), dtype="uint8")
    img = cv2.imdecode(image, -1)

    return img


def preprocessing(img):
    img = img[110:180, 0:636]
    return img


def main():
    img = url_to_image("https://ftp.lsf-wesel-rheinhausen.de/httpdocs/webcam-lsf/CLUB.jpg")
    img = preprocessing(img)
    cv2.imshow("processed", img)
    cv2.waitKey()
    pipeline = keras_ocr.pipeline.Pipeline()
    print(prediction_groups=pipeline.recognize(img))
# pipeline = keras_ocr.pipeline.Pipeline()
# return pipeline.recognize(img)
