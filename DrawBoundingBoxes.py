import cv2
import csv

annotations = []
drawing = False
x_start, y_start = 0, 0

# Draw the bounding boxes
def draw_box(event, x, y, flags, param):
    global x_start, y_start, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        drawing = False
        class_label = input("Enter the class label for this bounding box: ")
        annotations.append((image_path, x_start, y_start, x_end, y_end, class_label))
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow('Image', image)

# We can change this to the folder path later
image_path = 'gesture_test.jpeg'
image = cv2.imread(image_path)
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_box)

while True:
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

with open('annotations.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_path', 'x_start', 'y_start', 'x_end', 'y_end', 'class_label'])
    for annotation in annotations:
        writer.writerow(annotation)

print("Annotations saved to annotations.csv")
