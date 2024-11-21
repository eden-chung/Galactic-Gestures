import cv2
import csv
import os
from PIL import Image
import pillow_heif  # To convert HEIC to JPG
import time
import signal
import sys

annotations = []
drawing = False
x_start, y_start = 0, 0
current_annotation = None

# TODO: MODIFY THIS PART HERE
image_folder = 'right_shoot'  # MODIFY THIS
current_class = 'right_shoot'  # MODIFY THIS
label_mapping = {
    'left': 0,
    'left_shoot': 1,
    'shoot': 2,
    'right': 3,
    'right_shoot': 4
}
current_class_number = label_mapping[current_class]

output_csv = os.path.join(image_folder, 'annotations.csv')

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_path', 'x_start', 'y_start', 'x_end', 'y_end', 'class_label'])

# Still save to CSV in case we hit control C and exit
def handle_exit(signal, frame):
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(annotations)
    print(f"Annotations have been saved to {output_csv}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

# Function to draw the bounding boxes
def draw_box(event, x, y):
    global x_start, y_start, drawing, current_annotation, image, original_image

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        drawing = True

    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        drawing = False
        class_label = current_class_number

        # Overwrite with the new annotation
        current_annotation = (current_image_path, x_start, y_start, x_end, y_end, class_label)

        # Draw a new rectangle which overwrites the previous one
        image = original_image.copy()
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow('Image', image)

# If the image is in HEIC format, convert it to JPG
def convert_heic_to_jpg(heic_path):
    heif_file = pillow_heif.read_heif(heic_path)
    image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride
    )
    image = image.convert("RGB")
    jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
    image.save(jpg_path, format="JPEG", quality=95)
    time.sleep(0.1)  # wait for system to finish creating the new file
    return jpg_path

supported_formats = ['.jpg', '.jpeg', '.png', '.heic']
files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_formats]

# Go through all images in the dataset
for file in files:
    original_image_path = os.path.join(image_folder, file)
    current_image_path = original_image_path  # Initialize current_image_path

    # Check if HEIC
    if original_image_path.lower().endswith('.heic'):
        converted_jpg_path = convert_heic_to_jpg(original_image_path)
        if converted_jpg_path:
            print(f"Conversion of HEIC image {original_image_path} successful")
            try:
                os.remove(original_image_path)  # Remove HEIC file
                print(f"Deleted HEIC file: {original_image_path}")
                current_image_path = converted_jpg_path  # Update to JPG path
            except Exception as e:
                print(f"There is an error deleting HEIC file {original_image_path}: {e}")
                continue
        else:
            print(f"Conversion error with this image: {original_image_path}.")
            continue

    # Error checking
    if not os.path.exists(current_image_path):
        print(f"Can't find converted file: {current_image_path}")
        continue

    image = cv2.imread(current_image_path)
    if image is None:
        print(f"Can't read the image: {current_image_path}")
        continue

    original_image = image.copy()
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_box)

    current_annotation = None

    print(f"Processing: {current_image_path}")
    while True:
        cv2.imshow('Image', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # q to quit
            print("pressed q")
            handle_exit(None, None) # Save annotations and exit
        elif key == 13 or key == 10: # Enter key to go to next image
            print("pressed enter")
            if current_annotation:
                annotations.append(current_annotation) # Save only the latest box
                print(f"Saved annotation: {current_annotation}")

                # Add to the CSV
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(current_annotation)

            break

    cv2.destroyAllWindows()

print(f"Annotations saved to {output_csv}")
