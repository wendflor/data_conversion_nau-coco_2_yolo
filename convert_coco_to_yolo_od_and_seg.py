import json
import os
import shutil
import yaml

def convert2yolo(input_json_path, output_dir):

    # Extract the base name of the JSON file (without the extension)
    json_basename = os.path.splitext(os.path.basename(input_json_path))[0]
    
    # Define the directory where the source images are expected to be (same name as the JSON file)
    images_dir = os.path.join(os.path.dirname(input_json_path), json_basename)

    # Define the directories where the converted images and labels will be saved
    output_images_dir = os.path.join(output_dir, 'images')  # Directory to store images
    labels_dir = os.path.join(output_dir, 'labels')  # Directory to store labels
    bboxes_dir = os.path.join(labels_dir, 'bboxes')  # Directory for bounding box labels
    masks_dir = os.path.join(labels_dir, 'masks')  # Directory for segmentation masks

    # Create output directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Load COCO annotations from the input JSON file
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Function to convert bounding boxes to YOLO format
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2] / 2.0) * dw  # x center
        y = (box[1] + box[3] / 2.0) * dh  # y center
        w = box[2] * dw  # width
        h = box[3] * dh  # height
        return (x, y, w, h)

    # Function to convert segmentation data to YOLO format
    def convert_segmentation(size, segmentation):
        dw = 1. / size[0]
        dh = 1. / size[1]
        yolo_segmentation = []
        # Loop through each segment and convert the coordinates
        for segment in segmentation:
            yolo_segment = [coord * dw if i % 2 == 0 else coord * dh for i, coord in enumerate(segment)]
            yolo_segmentation.append(yolo_segment)
        return yolo_segmentation

    # Process each image in the COCO dataset
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        # Create corresponding YOLO label files for bounding boxes and masks
        bbox_filename = os.path.splitext(file_name)[0] + '.txt'  # Label file name
        bbox_filepath = os.path.join(bboxes_dir, bbox_filename)  # Bounding box file path
        mask_filepath = os.path.join(masks_dir, bbox_filename)  # Mask file path

        with open(bbox_filepath, 'w') as bbox_file, open(mask_filepath, 'w') as mask_file:
            # Process each annotation in the COCO dataset
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']

                    # Convert and save bounding boxes
                    if 'bbox' in annotation:
                        bbox = convert_bbox((width, height), annotation['bbox'])
                        bbox_file.write(f"{category_id} {' '.join(map(str, bbox))}\n")

                    # Convert and save segmentations
                    if 'segmentation' in annotation:
                        yolo_segmentation = convert_segmentation((width, height), annotation['segmentation'])
                        for seg in yolo_segmentation:
                            mask_file.write(f"{category_id} {' '.join(map(str, seg))}\n")

        # Copy the image to the output directory for images
        src_image_path = os.path.join(images_dir, file_name)
        dst_image_path = os.path.join(output_images_dir, file_name)
        shutil.copy(src_image_path, dst_image_path)

    print("Conversion and copying of images completed.")

def create_yaml_file(output_folder, json_path):
    # Load COCO JSON to extract category names for the YAML file
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Extract class names from the COCO data categories
    categories = coco_data.get('categories', [])
    class_names = [category['name'] for category in categories]
    num_classes = len(class_names)  # Total number of classes
    
    # Prepare the YAML data structure
    data = {
        'train': os.path.join(output_folder, 'train', 'images'),  # Path to training images
        'val': os.path.join(output_folder, 'val', 'images'),  # Path to validation images
        'test': os.path.join(output_folder, 'test', 'images'),  # Path to test images
        'nc': num_classes,  # Number of classes
        'names': class_names  # List of class names
    }

    # Save the YAML file in the output directory
    yaml_path = os.path.join(output_folder, 'data.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"YAML file created: {yaml_path}")

if __name__ == "__main__":
    # Convert the COCO train split to YOLO format
    input_json_path = 'red_truck_cab_simple_coco/train.json'  # Train dataset JSON
    output_dir = 'red_truck_cab_dataset_yolo/train'  # Output directory for YOLO train dataset
    convert2yolo(input_json_path, output_dir)

    # Convert the COCO validation split to YOLO format
    input_json_path = 'red_truck_cab_simple_coco/validation.json'  # Validation dataset JSON
    output_dir = 'red_truck_cab_dataset_yolo/val'  # Output directory for YOLO validation dataset
    convert2yolo(input_json_path, output_dir)

    # Convert the COCO test split to YOLO format
    input_json_path = 'red_truck_cab_simple_coco/test.json'  # Test dataset JSON
    output_dir = 'red_truck_cab_dataset_yolo/test'  # Output directory for YOLO test dataset
    convert2yolo(input_json_path, output_dir)

    # Create the YAML file for the dataset configuration
    output_folder = os.path.dirname(output_dir)
    create_yaml_file(output_folder, input_json_path)

