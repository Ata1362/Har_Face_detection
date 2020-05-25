import cv2 as cv

face_data = "F:\Coding Projects\Visual Studio PYTHON\Har_Face_detection\XML\haarcascade_frontalface_default.xml"

face_detect = cv.CascadeClassifier(face_data)

cam = cv.VideoCapture(1)

while True:
    ret, frame = cam.read()
    frame2 = frame.copy()
    frame1 = frame.copy()
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(frame2, 1.2, 5, 10, (50, 50), (250, 250))
    for (x, y, z, w) in faces:
        cv.rectangle(frame1, (x, y), (x+z, y+w), (0, 255, 0), 4)
        text = 'Face' + str(x)
        cv.putText(frame1, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv.LINE_AA)
    frame1 = cv.addWeighted(frame, 0.8, frame1, 0.2, 0)
    cv.imshow('faces', frame1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()