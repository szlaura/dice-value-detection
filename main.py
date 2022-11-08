import cv2 as cv
import numpy as np


def main():
    file_name = "dice04.jpg"
    src = cv.imread(file_name, cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 300

    params.filterByCircularity = True
    params.minCircularity = 0.2

    params.filterByConvexity = True
    params.minConvexity = 0.2

    params.filterByInertia = True
    params.minInertiaRatio = 0.014
    params.maxInertiaRatio = 0.4

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(gray)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv.drawKeypoints(gray, keypoints, blank, (0, 0, 255),
                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    text = "Dots number: " + str(len(keypoints))
    print(text)

    cv.putText(blobs, text, (3, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Rolling dice", blobs)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return 0


if __name__ == "__main__":
    main()