import cv2
from cv2 import aruco
import numpy as np
import os


def image_augmentation(frame, src_image, dst_points):
    src_h, src_w = src_image.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
    warp_image = cv2.warpPerspective(src_image, H, (frame_w, frame_h))
    # cv2.imshow("warp image", warp_image)
    cv2.fillConvexPoly(mask, dst_points, 255)
    results = cv2.bitwise_and(warp_image, warp_image, frame, mask=mask)


def read_images(dir_path):
    img_list = []
    files = os.listdir(dir_path)
    for file in files:
        img_path = os.path.join(dir_path, file)
        image = cv2.imread(img_path)
        img_list.append(image)
    return img_list


marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

images_list = read_images("AR_BASIC/images")

cap = cv2.VideoCapture("AR_BASIC/images/marker.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):

            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            if ids[0] <= 12:
                image_augmentation(frame, images_list[ids[0]], corners)
            else:
                image_augmentation(frame, images_list[0], corners)

    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()