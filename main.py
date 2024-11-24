import pathlib
import cv2


# Loading the face detection model
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)


if not camera.isOpened():
    print("application could not open the camera")
    exit()
    

while True:
    
    ret, frame = camera.read()
    
    if frame is not None:
        print("Frame shape:", frame.shape)
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draws the rectangles around detected faces

    for (x, y, width, height) in faces :
        cv2.rectangle(frame ,(x,y), (x + width, y + height), (255, 255, 0),2)
    
    # Display the frame with detected faces
    cv2.imshow("Faces", frame)
    
    # exit the loop if q is pressed 
    if cv2.waitKey(1) == ord("q") :
        break
    
    
camera.release()
cv2.destroyAllWindows()


