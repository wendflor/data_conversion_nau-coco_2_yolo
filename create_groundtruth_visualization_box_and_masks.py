import os
import cv2
import yaml
import numpy as np

def load_bbox_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls, x_center, y_center, width, height = map(float, parts[:5])
        labels.append({
            'cls': int(cls),
            'bbox': [x_center, y_center, width, height],
            'mask_coords': []
        })
    
    return labels

def load_mask_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        mask_coords = [float(coord) for coord in parts[1:]]  # Rest are mask coordinates
        labels.append({
            'cls': cls,
            'bbox': [],
            'mask_coords': mask_coords
        })
    
    return labels

def draw_labels(image, bbox_labels, mask_labels, class_names):
    h, w = image.shape[:2]
    overlay = image.copy()

    for label in bbox_labels:
        cls = label['cls']
        bbox = label['bbox']

        # Convert YOLO format to (x1, y1, x2, y2)
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add the class label
        label_text = class_names[cls]  # Correctly index into class_names
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    for label in mask_labels:
        cls = label['cls']
        mask_coords = label['mask_coords']

        # Draw the mask
        if mask_coords:
            mask_coords = np.array(mask_coords).reshape(-1, 2)
            mask_coords[:, 0] *= w
            mask_coords[:, 1] *= h
            mask_coords = mask_coords.astype(np.int32)
            cv2.fillPoly(overlay, [mask_coords], color=(0, 255, 0))  # Green color

            # Add the class label (for mask, place the label in the centroid of the mask)
            centroid = np.mean(mask_coords, axis=0).astype(int)
            label_text = class_names[cls]  # Correctly index into class_names
            cv2.putText(image, label_text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend the original image with the overlay
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image


def process_dataset(images_dir, labels_base_dir, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    bboxes_dir = os.path.join(labels_base_dir, 'bboxes')
    masks_dir = os.path.join(labels_base_dir, 'masks')

    for img_name in os.listdir(images_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, img_name)
            bbox_path = os.path.join(bboxes_dir, os.path.splitext(img_name)[0] + '.txt')
            mask_path = os.path.join(masks_dir, os.path.splitext(img_name)[0] + '.txt')

            image = cv2.imread(img_path)
            bbox_labels = load_bbox_labels(bbox_path)
            mask_labels = load_mask_labels(mask_path)

            image = draw_labels(image, bbox_labels, mask_labels, class_names)
            output_path = os.path.join(output_dir, img_name)
            cv2.imwrite(output_path, image)

if __name__ == "__main__":
    with open('data.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    datasets = ['train', 'val', 'test']
    base_dir = os.path.dirname(config['train'])
    class_names = config['names']
    output_base_dir = 'output'

    for dataset in datasets:
        images_dir = config[dataset]
        labels_base_dir = os.path.join(os.path.dirname(images_dir), 'labels')
        output_dir = os.path.join(os.path.dirname(images_dir), 'check_groundtruth')
        process_dataset(images_dir, labels_base_dir, class_names, output_dir)
