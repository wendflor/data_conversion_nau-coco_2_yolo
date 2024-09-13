import os
import cv2
import yaml
import numpy as np

def load_labels(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        cls, x_center, y_center, width, height = map(float, parts[:5])
        mask_coords = [float(coord) for coord in parts[5:]]  # Rest are mask coordinates
        labels.append({
            'cls': int(cls),
            'bbox': [x_center, y_center, width, height],
            'mask_coords': mask_coords
        })
    
    return labels

def draw_labels(image, labels, class_names):
    h, w = image.shape[:2]
    overlay = image.copy()

    for label in labels:
        cls = label['cls']
        bbox = label['bbox']
        mask_coords = label['mask_coords']

        # Convert YOLO format to (x1, y1, x2, y2)
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the mask
        if mask_coords:
            mask_coords = np.array(mask_coords).reshape(-1, 2)
            mask_coords[:, 0] *= w
            mask_coords[:, 1] *= h
            mask_coords = mask_coords.astype(np.int32)
            cv2.fillPoly(overlay, [mask_coords], color=(0, 255, 0))  # Green color

        # Add the class label
        label_text = f"{class_names[cls]}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend the original image with the overlay
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image

def process_dataset(images_dir, labels_dir, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
        
        if not os.path.exists(label_path):
            print(f"No label found for {img_path}")
            continue
        
        image = cv2.imread(img_path)
        labels = load_labels(label_path)
        image_with_labels = draw_labels(image, labels, class_names)
        
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image_with_labels)
        print(f"Processed and saved: {output_path}")

def main():
    with open('data.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
    
    class_names = data_config['names']
    
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
        images_dir = data_config[dataset]
        labels_dir = images_dir.replace('images', 'labels')
        output_dir = os.path.join(os.path.dirname(images_dir), 'check_groundtruth')

        process_dataset(images_dir, labels_dir, class_names, output_dir)

if __name__ == '__main__':
    main()
