import cv2
from cv2 import aruco
import numpy as np
import os

calib_data_path = "./augmented-reality/calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

# ar text: 로봇비전
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

MARKER_SIZE = 5.1  # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

images_list = read_images("./augmented-reality/AR_BASIC/images")

cap = cv2.VideoCapture("./augmented-reality/AR_BASIC/images/marker.mp4")
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv2.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_left = corners[0].ravel()
            top_right = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            
            if ids[0] <= 12:
                image_augmentation(frame, images_list[ids[0]], corners)
            else:
                image_augmentation(frame, images_list[0], corners)
            
            # draw cube
            cv2.line(frame, top_left,top_right,(255,255,0),3)
            cv2.line(frame, bottom_left,bottom_right,(255,255,0),3)
            cv2.line(frame, top_left,bottom_left,(255,255,0),3)
            cv2.line(frame, top_right,bottom_right,(255,255,0),3)
            
            cv2.line(frame, (top_left[0],top_left[1]+100),(top_right[0],top_right[1]+100),(255,255,0),3)
            cv2.line(frame, (bottom_left[0],bottom_left[1]+100),(bottom_right[0],bottom_right[1]+100),(255,255,0),3)
            cv2.line(frame, (top_left[0],top_left[1]+100),(bottom_left[0],bottom_left[1]+100),(255,255,0),3)
            cv2.line(frame, (top_right[0],top_right[1]+100),(bottom_right[0],bottom_right[1]+100),(255,255,0),3)

            cv2.line(frame, top_left,(top_left[0],top_left[1]+100),(255,255,0),3)
            cv2.line(frame, top_right,(top_right[0],top_right[1]+100),(255,255,0),3)
            cv2.line(frame, bottom_left,(bottom_left[0],bottom_left[1]+100),(255,255,0),3)
            cv2.line(frame, bottom_right,(bottom_right[0],bottom_right[1]+100),(255,255,0),3)
            
    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
    out.write(frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()