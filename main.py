import cv2 as cv
import numpy as np


def main():
    file_name = 'dice02.png'


    src = cv.imread(file_name, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        return -1

    dim = 360
    src = cv.resize(src, (dim, dim), interpolation=cv.INTER_AREA)

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)

    rows = gray.shape[0]
    dots = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 16,
                              param1=100, param2=20,
                              minRadius=1, maxRadius=20)

    if dots is not None:
        dots = np.uint16(np.around(dots))
        for coords in dots[0, :]:
            center, radius = (coords[0], coords[1]), coords[2]
            cv.circle(src, center, radius, (0, 255, 0), -1)
        print("Number of detected dice dots: ", len(dots[0, :]))

    cv.imshow("detected dice dots", src)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main()