import keras_ocr
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
from numpy.core.defchararray import capitalize


def list_conversion(prediction_groups):
    print(prediction_groups)
    liste = list()
    tempListe = list()
    tempListe2 = list()
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
        #print(i[:1].capitalize() + '-' + i[1:])
        tempListe2.append(i[:1].capitalize() + '-' + i[1:])
    vergleichs_liste = ['D-9524', 'D-2244', 'D-0158', 'D-1619', 'D-7921', 'D-2860']
    if vergleichs_liste == tempListe2:
        print("Success!")
    else:
        print("Fail!")
    return tempListe2


def image_recognition(input_images):
    pipeline = keras_ocr.pipeline.Pipeline()
    return list_conversion(prediction_groups=pipeline.recognize(input_images))
