import time

import cv2
import pytesseract


def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./preprocess/img_gray.png", img)
    return img


# blur
def blur(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(r"./preprocess/img_blur.png", img)
    return img_blur


# threshold
def threshold(img):
    # pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(r"./preprocess/img_threshold.png", img)
    return img


def contours_text(orig, contours):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Drawing a rectangle on copied image
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow('cnt', rect)
        cv2.waitKey()

        # Cropping the text block for giving input to OCR
        cropped = orig[y:y + h, x:x + w]

        # Apply OCR on the cropped image
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped)

        print(text)


def main():
    # Finding contours
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    cam = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
    cam.set(3, 1920)
    cam.set(4, 1080)
    time.sleep(2)
    while True:
        ret_val, img = cam.read()
        im_gray = gray(img)
        im_blur = blur(im_gray)
        im_thresh = threshold(im_blur)
        text = pytesseract.pytesseract.image_to_string(im_thresh)
        print("Detected text: " + text)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
