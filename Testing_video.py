import cv2

for i in range(2):  
    print(f"Testing camera index {i}")
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"Camera index {i} not available.")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera index {i} is not capturing frames.")
            break
        
        cv2.imshow(f"Camera Index {i}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
cv2.destroyAllWindows()