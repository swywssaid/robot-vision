# Augmented Reality <!-- omit in toc -->

AR 실행결과와 주요 코드입니다. 각 헤더에 링크가 걸려있습니다.

<br>

- [1. Generate Markers](#1-generate-markers)
  - [1.1. Code](#11-code)
- [2. Camera Calibration](#2-camera-calibration)
  - [2.1. Code](#21-code)
- [3. AR Distance](#3-ar-distance)
  - [3.1. Result](#31-result)
  - [3.2. Code](#32-code)
- [4. AR Basic](#4-ar-basic)
  - [4.1. Result](#41-result)
  - [4.2. Code](#42-code)
- [5. AR 3D](#5-ar-3d)
  - [5.1. Result](#51-result)
  - [5.2. Code](#52-code)

<br><br>

## 1. [Generate Markers](https://github.com/swywssaid/robot-vision/tree/main/augmented-reality/GENERATE_MARKERS)
aruco 라이브러리를 이용하여 ArUco Marker를 생성합니다.
### 1.1. [Code](https://github.com/swywssaid/robot-vision/blob/main/augmented-reality/GENERATE_MARKERS/main.py#L10)
```python
marker_image = aruco.drawMarker(marker_dict, id, MARKER_SIZE)
```
<br><br>

## 2. [Camera Calibration](https://github.com/swywssaid/robot-vision/tree/main/augmented-reality/CAMERA_CALIBRATION)
Camera Calibration를 진행하는 코드입니다. 체커보드를 기준으로 수행하며 체커보드의 종류와 칸의 크기가 필요합니다. 
### 2.1. [Code](https://github.com/swywssaid/robot-vision/blob/main/augmented-reality/CAMERA_CALIBRATION/camera_calibration.py#L42)
```python
ret, corners = cv.findChessboardCorners(image, CHESS_BOARD_DIM, None)
if ret == True:
    obj_points_3D.append(obj_3D)
    corners2 = cv.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
    img_points_2D.append(corners2)

    img = cv.drawChessboardCorners(image, CHESS_BOARD_DIM, corners2, ret)
```
<br><br>

## 3. [AR Distance](https://github.com/swywssaid/robot-vision/tree/main/augmented-reality/AR_DISTANCE)
x,y,z 좌표축을 그리고 마커와 카메라 간의 거리를 측정합니다.
### 3.1. Result
<img src="augmented-reality/AR_DISTANCE/AR_DISTANCE_RESULT.gif" width="400" height="350">

<br>

### 3.2. [Code](https://github.com/swywssaid/robot-vision/blob/main/augmented-reality/AR_DISTANCE/main.py#L55)
```python
# Calculating the distance
distance = np.sqrt(
    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
)
# Draw the pose of the marker
point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
```

<br><br>

## 4. [AR Basic](https://github.com/swywssaid/robot-vision/tree/main/augmented-reality/AR_BASIC)
각 마커에 매칭되는 이미지로 마커를 매핑합니다.
### 4.1. Result
<img src="augmented-reality/AR_BASIC/AR_BASIC_RESULT.gif" width="400" height="350">

<br>

### 4.2. [Code](https://github.com/swywssaid/robot-vision/blob/main/augmented-reality/AR_BASIC/main.py#L7)
```python
# mapping markers and images
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
```
<br><br>

## 5. [AR 3D](https://github.com/swywssaid/robot-vision/tree/main/augmented-reality/AR_3D)
마커를 기준으로 정육면체를 그립니다.
### 5.1. Result
<img src="augmented-reality/AR_3D/AR_3D_RESULT.gif" width="400" height="350">

<br>

### 5.2. [Code](https://github.com/swywssaid/robot-vision/blob/main/augmented-reality/AR_3D/main.py#L78)
```python
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
```