import os
import cv2
import yaml
import numpy as np

def load_bbox_labels(file_path):
    # Open the bounding box labels file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse each line to extract class and bounding box coordinates
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls, x_center, y_center, width, height = map(float, parts[:5])
        labels.append({
            'cls': int(cls),  # Convert class to an integer
            'bbox': [x_center, y_center, width, height],  # YOLO format bbox
            'mask_coords': []  # Initialize empty mask coordinates
        })
    
    return labels

def load_mask_labels(file_path):
    # Open the mask labels file and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse each line to extract class and mask coordinates
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])  # The class of the object
        mask_coords = [float(coord) for coord in parts[1:]]  # The mask coordinates
        labels.append({
            'cls': cls,
            'bbox': [],  # Initialize empty bbox
            'mask_coords': mask_coords  # Store mask coordinates
        })
    
    return labels

def draw_labels(image, bbox_labels, mask_labels, class_names):
    # Get image dimensions
    h, w = image.shape[:2]
    overlay = image.copy()

    # Loop through bounding box labels and draw them on the image
    for label in bbox_labels:
        cls = label['cls']
        bbox = label['bbox']

        # Convert YOLO bbox format to (x1, y1, x2, y2)
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add the class label on the image
        label_text = class_names[cls]  # Get class name by index
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Loop through mask labels and draw them on the image
    for label in mask_labels:
        cls = label['cls']
        mask_coords = label['mask_coords']

        # Draw the mask if mask coordinates exist
        if mask_coords:
            mask_coords = np.array(mask_coords).reshape(-1, 2)
            mask_coords[:, 0] *= w  # Scale x-coordinates to image width
            mask_coords[:, 1] *= h  # Scale y-coordinates to image height
            mask_coords = mask_coords.astype(np.int32)  # Convert to integer
            cv2.fillPoly(overlay, [mask_coords], color=(0, 255, 0))  # Draw mask with green color

    # Blend the original image with the overlay containing the masks
    alpha = 0.5  # Transparency factor for the overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image

def process_dataset(images_dir, labels_base_dir, class_names, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths for bounding boxes and masks
    bboxes_dir = os.path.join(labels_base_dir, 'bboxes')
    masks_dir = os.path.join(labels_base_dir, 'masks')

    # Loop through all images in the directory
    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_name)
            bbox_path = os.path.join(bboxes_dir, os.path.splitext(img_name)[0] + '.txt')
            mask_path = os.path.join(masks_dir, os.path.splitext(img_name)[0] + '.txt')

            # Load image
            image = cv2.imread(img_path)

            # Load bounding box and mask labels
            bbox_labels = load_bbox_labels(bbox_path)
            mask_labels = load_mask_labels(mask_path)

            # Draw the labels on the image
            image = draw_labels(image, bbox_labels, mask_labels, class_names)

            # Save the annotated image in the output directory
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Load configuration from data.yaml
    with open('data.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Define the datasets to process (train, val, test)
    datasets = ['train', 'val', 'test']

    # Get the base directory of the dataset and class names
    base_dir = os.path.dirname(config['train'])
    class_names = config['names']

    # Process each dataset (train, val, test)
    for dataset in datasets:
        images_dir = config[dataset]  # Get image directory for the dataset
        labels_base_dir = os.path.join(os.path.dirname(images_dir), 'labels')  # Define label directory
        output_dir = os.path.join(os.path.dirname(images_dir), 'check_groundtruth')  # Define output directory
        process_dataset(images_dir, labels_base_dir, class_names, output_dir)  # Process the dataset
