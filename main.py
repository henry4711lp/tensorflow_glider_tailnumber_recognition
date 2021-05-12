import matplotlib.pyplot as plt
import cv2
import keras_ocr
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
from numpy.core.defchararray import capitalize

pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
images = [
    keras_ocr.tools.read(url) for url in [
        r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider1.jpeg',
        r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider2.jpeg',
        r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider3.jpeg',
        r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider4.jpeg'
    ]
]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
liste = list()
tempListe = list()
prediction_groups = pipeline.recognize(images)
for i in prediction_groups:
    for a in i:
        liste.append(a[0])
for i in liste:
    if len(i) == 5:
        tempListe.append(i)
    if len(i) == 6:
        tempListe.append(i[:1] + i[2:])
liste = tempListe

for i in liste:
    print(i[:1].capitalize() + '-' + i[1:])
