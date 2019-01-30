import cv2

# Read the first frame of the video
video = cv2.VideoCapture(0)

# Create a Cascade Classifier object
face_cascade = cv2.CascadeClassifier("/Users/arjungurung/Desktop/Master OpenCV/Haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/arjungurung/Desktop/Master OpenCV/Haarcascades/haarcascade_eye.xml")

# Initializer to record the amount of frames
a = 1
while True:
    a = a+1

    # check is a boolean that returns true if python was able to read the video capture object
    # frame is the numpy array that represents the first image that the video captures
    check, frame = video.read()

    # print(frame), if you'd like to see the stored arrays or frames

    # Works best on gray scale mode
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Search the coordinates of the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, w, h in faces:
        # print(x, y, w, h) Checking the values of the points
        # x = x-axis, y = y-axis, w = length and h = breadth of the rectangle

        # cv2.rectangle(image, point1, point2, color, thickness)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Since the eyes are in the face, we reduce the sample size for the eye detector
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(image, point1, point2, color, thickness)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    cv2.imshow('Capturing', frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(a)
video.release()

cv2.destroyAllWindows()