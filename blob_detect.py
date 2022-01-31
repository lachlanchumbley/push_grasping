# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BlobDetector:
    def __init__(self, writeImages=True, showImages=True, cv2Image=False):

        self.bridge = CvBridge()
        self.state = Quadrant.INIT
        self.closest_state = Quadrant.INIT
        self.angle = None
        self.angular_velocity = 0
        self.calc_time = None
        self.writeImages = writeImages
        self.showImages = showImages
        self.cv2Image = cv2Image

    def find_center(self, im):
        # load the image
        if not self.cv2Image:
            im = self.bridge.imgmsg_to_cv2(im, desired_encoding="8UC3")

        result = im.copy()
        image = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(image, light_orange, dark_orange)
        result = cv2.bitwise_and(image, image, mask=mask)

        if self.showImages:
            cv2.imshow("orig", im)
            cv2.imshow("hsv", image)
            cv2.imshow("mask", mask)
            cv2.waitKey(1)

        if self.writeImages:
            cv2.imwrite("img_temp.jpeg", im)
            cv2.imwrite("mask.jpeg", mask)
            cv2.imwrite("result.jpeg", result)

        contours = None
        if PYTHON3:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # print(tmp, contours)
            contours = sorted(
                contours, key=lambda el: cv2.contourArea(el), reverse=True
            )

        else:
            _, contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours.sort(key=lambda el: cv2.contourArea(el), reverse=True)

        canvas = result.copy()

        M = cv2.moments(contours[0])
        center1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(canvas, center1, 2, (0, 255, 0), -1)

        cv2.waitKey(0)

    
if __name__ == "__main__":
    try:
        angle_class = BlobDetectorService()
    except KeyboardInterrupt:
        pass