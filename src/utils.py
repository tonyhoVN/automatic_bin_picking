import cv2 as cv
from math import atan2, degrees
import numpy as np
from scipy.spatial.transform import Rotation as R


markerLength = 0.05
object_point = np.zeros((4,3))

object_point[0,:] = [markerLength/2.0, markerLength/2.0, 0] 
object_point[1,:] = [-markerLength/2.0, markerLength/2.0, 0] 
object_point[2,:] = [-markerLength/2.0, -markerLength/2.0, 0] 
object_point[3,:] = [markerLength/2.0, -markerLength/2.0, 0] 


camera_matrix = np.array([[913.60406494,   0.         ,645.53533936],
						  [  0.,         913.78271484, 363.54202271],
						  [  0.,           0.,           1.        ]])

distorsion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

Rot_reverse = np.array([[-1, 0, 0],[0,1,0],[0,0,-1]])

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
arucoPara = cv.aruco.DetectorParameters()
arucoDetect = cv.aruco.ArucoDetector(arucoDict, arucoPara)

def aruco_display(image):

	corners, ids, rejected = arucoDetect.detectMarkers(image)
	angle_avg = 0
	rot_avg = np.eye(3)
	count = 0
	rvecs = []
	tvecs = []

	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)

			corners_ = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners_
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# cv.line(image, topLeft, topRight, (255, 0, 0), 2)
			# cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
			# cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			# cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			cv.line(image, bottomRight, bottomLeft,(0, 0, 255), 2)
			cv.line(image, bottomRight, topRight,(0, 255, 0), 2)

			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv.putText(image, str(markerID),(topLeft[0], topLeft[1] - 20), cv.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			# print("[Inference] ArUco marker ID: {}".format(markerID))
			
			# Calculate rotation angle 
			if markerID == 10 or markerID ==1:
				angle_avg = degrees(atan2(bottomLeft[1]-bottomRight[1], bottomLeft[0]-bottomRight[0]))

				# Pose estimation 
				n,rvec,tvec = cv.solvePnP(object_point, markerCorner, camera_matrix, distorsion, False, cv.SOLVEPNP_ITERATIVE)
				image = cv.drawFrameAxes(image, camera_matrix, distorsion, rvec, tvec,0.1)
				# image = cv.aruco.drawDetectedMarkers(image,markerCorner,markerID)
				# pose = cv.aruco.estimatePoseSingleMarkers(markerCorner,markerLength, camera_matrix,distorsion)
				rot_matrix_test, _ = cv.Rodrigues(rvec)
				rot_matrix = R.from_euler('z', angle_avg, degrees=True).as_matrix()
				rot_matrix_down = np.matmul(Rot_reverse,rot_matrix)				
				# print("Row matrix test: ")
				# print(rot_matrix_test)
				# print("Measure rot matrix: ")
				# print(rot_matrix)
				rot_avg = rot_matrix_down

			count += 1

		# angle_avg = angle_avg/count
		# rot_avg = rot_avg/count
		
	return image, angle_avg, rot_avg




