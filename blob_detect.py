# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        # load the image
        path = "/Users/lachlanchumbley/Documents/SRP/right.jpg"
        raw_image = cv2.imread(path)

        rgb_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(rgb_image)
        # plt.show()

        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        # rgba(227,120,64,255)
        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(hsv_image, light_orange, dark_orange)
        result = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

        binary_result = np.where(result > 0, 1, 0)


        # show the images
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.show()

        # calculate moments of binary image
        # M = cv2.moments(binary_result)
        #
        # # calculate x,y coordinate of center
        # cX = int(M["m10"] / M["m00"])
        # cY = int(M["m01"] / M["m00"])
        #
        # # put text and highlight the center
        # cv2.circle(rgb_image, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(rgb_image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #
        # # display the image
        # cv2.imshow("Image", rgb_image)
        # cv2.waitKey(0)

    except KeyboardInterrupt:
        pass