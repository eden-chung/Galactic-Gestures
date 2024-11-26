import cv2

cap = cv2.VideoCapture(0)  # Use default camera

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture a frame.")
        break

    cv2.imshow("Live Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q'
        break

cap.release()
cv2.destroyAllWindows()