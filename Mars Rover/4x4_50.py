import cv2
import cv2.aruco as aruco

VideoCap = False   #if true we can detect markers using web cam
cap = cv2.VideoCapture(0)

def findAruco(img, marker_size=4, total_markers=50, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    arucoDict = aruco.Dictionary_get(key)  # Use the specified marker size and total markers
    arucoparam = aruco.DetectorParameters_create()
    bbox, ids, _ = aruco.detectMarkers(gray, arucoDict, parameters=arucoparam)

    if ids is not None and draw:
        for i in range(len(ids)):
            # Get the corners of the detected marker
            corners = bbox[i][0]

            # Calculate the center of the marker
            center_x = int(sum(corners[:, 0]) / 4)
            center_y = int(sum(corners[:, 1]) / 4)

            # Draw the ID on the image
            cv2.putText(img, str(ids[i][0]), (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        aruco.drawDetectedMarkers(img, bbox)

    return bbox, ids

while True:
    if VideoCap:
        _, img = cap.read()
    else:
        img = cv2.imread("Resources/comp.jpeg")
        img = cv2.resize(img, (0, 0), fx=1, fy=1)

    bbox, ids = findAruco(img, marker_size=4, total_markers=50)  # Use 4x4_50 markers

    if cv2.waitKey(1) == 113:
        break

    cv2.imshow("img", img)

cv2.destroyAllWindows()
