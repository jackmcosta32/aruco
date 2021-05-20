import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Utils

def detect_and_replace_aruco(dict, frame, replacement_img):
    """
    Detects and replaces an ArUco form a certain dictionary with an specified image.
    :param dict: dict ArUco dictionary
    :param frame: list Destiny image
    :param replacement_img: list Source image
    :param debug: bool Enables debug mode.
    :return:
    """

    # Param definition
    output_img = frame

    # Initialize the aruco detector with it's default values
    detector_params = cv.aruco.DetectorParameters_create()

    # Extracts information from the replacement image
    [l, w, ch] = np.shape(replacement_img)
    src_pts = np.array([[0, 0], [w, 0], [w, l], [0, l]])

    # Detects the image markers
    marker_corners, marker_ids, rejected_corners = cv.aruco.detectMarkers(frame, dict, parameters=detector_params)

    # REMOVER
    # if len(marker_corners) == 0:
    #     marker_corners = rejected_corners

    detections_img = cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    # Prepare for image replacement
    for corner in marker_corners:
        # Set's the coordinates of the aruco
        destiny_pts = np.array(corner[0])

        # Calculates the homography
        homography, status = cv.findHomography(
            srcPoints=src_pts,
            dstPoints=destiny_pts
        )

        # Warps the source image it's destiny
        warped_image = cv.warpPerspective(
            src=replacement_img,
            M=homography,
            dsize=(frame.shape[1], frame.shape[0])
        )

        # Prepares the mask for occluding the aruco at the frame
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
        cv.fillConvexPoly(mask, np.int32([destiny_pts]), (255, 255, 255), cv.LINE_AA)

        # Erode the mask to not copy the boundary effects from the warping
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask = cv.erode(mask, element, iterations=3)

        # Copy the mask into 3 channels.
        warped_image = warped_image.astype(float)
        channeled_mask = np.zeros_like(warped_image)

        for i in range(0, ch):
            channeled_mask[:, :, i] = mask / 255

        # Masks the warped image
        masked_warped_image = cv.multiply(warped_image, channeled_mask)

        # Masks the frame than adds the masked frame to the output image
        masked_frame = cv.multiply(output_img.astype(float), 1 - channeled_mask)
        output_img = cv.add(masked_warped_image, masked_frame)

    # Casts image to uint8
    output_img = output_img.astype(np.uint8)

    return output_img, detections_img


# Definitions

replacement_img = cv.imread('./public/img/replacement/rep1.png')

# Prepare ArUco detection
dict_aruco = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate marker
marker_img = np.zeros((200, 200), dtype=np.uint8)
marker_img = cv.aruco.drawMarker(dict_aruco, 10, 200, marker_img, 1)

cv.imwrite("./public/img/ArUco/marker.png", marker_img)

# Prepare video capture
capture = cv.VideoCapture(0)

while True:
    # Capture frame
    _, frame = capture.read()

    # Detects and replaces the ArUco's with the correspondent image
    processed_frame, _ = detect_and_replace_aruco(dict_aruco, frame, replacement_img)

    cv.imshow('Video', processed_frame)

    # Release the video capture
    if cv.waitKey(30) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
