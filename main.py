import keras_ocr

import image_recognition
import webcam_analysis


def demo():
    images = [
        keras_ocr.tools.read(url) for url in [
            r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider1.jpeg',
            r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider2.jpeg',
            r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider3.jpeg',
            r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider4.jpeg',
            r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider5.jpeg',
            r'C:\Users\janse\PycharmProjects\pythonProject\resources\glider6.jpeg'
        ]
    ]

    webcam = [
        keras_ocr.tools.read(url) for url in [
            'https://ftp.lsf-wesel-rheinhausen.de/httpdocs/webcam-lsf/CLUB.jpg',
            'https://ftp.lsf-wesel-rheinhausen.de/httpdocs/webcam-lsf/START09.jpg',
            'https://ftp.lsf-wesel-rheinhausen.de/httpdocs/webcam-lsf/START27.jpg'
        ]
    ]
    print("Local Stream!: ")
    #print(image_recognition.image_recognition(images))
    print("Webcam Stream!: ")
    print(webcam_analysis.main())
    return 0;


def main():
    demo()


if __name__ == '__main__':
    main()
