import cv2
import csv
import os
from PIL import Image
import pillow_heif
import time
import signal
import sys

annotations = []
drawing = False
x_start, y_start = 0, 0
current_annotation = None

# TODO: MODIFY THIS PART HERE
image_folder = 'shoot_redo'  # MODIFY THIS
current_class = 'shoot'  # MODIFY THIS
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

def handle_exit(signal, frame):
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(annotations)
    print(f"Annotations have been saved to {output_csv}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

def draw_box(event, x, y, flags, param):
    global x_start, y_start, drawing, current_annotation, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            display_image = original_image.copy()
            cv2.rectangle(display_image, (x_start, y_start), (x, y), (255, 0, 0), 2)
            cv2.imshow('Image', display_image)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(display_image, (x_start, y_start), (x, y), (255, 0, 0), 2)
        cv2.imshow('Image', display_image)
        current_annotation = (current_image_path, x_start, y_start, x, y, current_class_number)

def convert_heic_to_jpg(heic_path):
    heif_file = pillow_heif.read_heif(heic_path)
    image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride
    )
    image = image.convert("RGB")
    jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
    image.save(jpg_path, format="JPEG", quality=95)
    while not os.path.exists(jpg_path):
        time.sleep(0.01)
    return jpg_path

supported_formats = ['.jpg', '.jpeg', '.png', '.heic']
files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1].lower() in supported_formats]

for file in files:
    original_image_path = os.path.join(image_folder, file)
    current_image_path = original_image_path

    if original_image_path.lower().endswith('.heic'):
        converted_jpg_path = convert_heic_to_jpg(original_image_path)
        if converted_jpg_path:
            print(f"Conversion of HEIC image {original_image_path} successful")
            try:
                os.remove(original_image_path)
                print(f"Deleted HEIC file: {original_image_path}")
                current_image_path = converted_jpg_path
            except Exception as e:
                print(f"There is an error deleting HEIC file {original_image_path}: {e}")
                continue
        else:
            print(f"Conversion error with this image: {original_image_path}.")
            continue

    if not os.path.exists(current_image_path):
        print(f"Can't find converted file: {current_image_path}")
        continue

    image = cv2.imread(current_image_path)
    if image is None:
        print(f"Can't read the image: {current_image_path}")
        continue

    h, w = image.shape[:2]
    max_width = 1280
    max_height = 720
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    original_image = image.copy()
    display_image = original_image.copy()

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_box)

    current_annotation = None

    print(f"Processing: {current_image_path}")
    while True:
        cv2.imshow('Image', display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("pressed q")
            handle_exit(None, None)
        elif key == 13 or key == 10:
            print("pressed enter")
            if current_annotation:
                # Get the coordinates rescaled since we resized the image
                scale_x = w / image.shape[1]
                scale_y = h / image.shape[0]
                
                x_start_scaled = int(current_annotation[1] * scale_x)
                y_start_scaled = int(current_annotation[2] * scale_y)
                x_end_scaled = int(current_annotation[3] * scale_x)
                y_end_scaled = int(current_annotation[4] * scale_y)

                scaled_annotation = (
                    current_annotation[0], # This is the image path
                    x_start_scaled,
                    y_start_scaled,
                    x_end_scaled,
                    y_end_scaled,
                    current_annotation[5] # This is the label
                )

                annotations.append(scaled_annotation)
                print(f"Saved annotation: {scaled_annotation}")

                # Write to CSV
                with open(output_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(scaled_annotation)

            break

    cv2.destroyAllWindows()

print(f"Annotations saved to {output_csv}")
