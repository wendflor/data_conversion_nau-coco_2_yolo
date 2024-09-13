import json
import os
import shutil
import yaml

def convert2yolo(input_json_path, output_dir):

    # Extract the name of the JSON file without the extension
    json_basename = os.path.splitext(os.path.basename(input_json_path))[0]
    images_dir = os.path.join(os.path.dirname(input_json_path), json_basename)  # source-image directory is expected to have the same name as .json-file

    output_images_dir = os.path.join(output_dir, 'images')  # Directory to copy the images to
    labels_dir = os.path.join(output_dir, 'labels')  # Directory to save the labels to
    bboxes_dir = os.path.join(labels_dir, 'bboxes')  # Directory to save the bounding boxes to
    masks_dir = os.path.join(labels_dir, 'masks')  # Directory to save the masks to

    # Create the output directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(bboxes_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Load COCO annotations
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Function to convert the bounding box to YOLO format
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2] / 2.0) * dw
        y = (box[1] + box[3] / 2.0) * dh
        w = box[2] * dw
        h = box[3] * dh
        return (x, y, w, h)

    # Function to convert the segmentation to YOLO format
    def convert_segmentation(size, segmentation):
        dw = 1. / size[0]
        dh = 1. / size[1]
        yolo_segmentation = []
        for segment in segmentation:
            yolo_segment = [coord * dw if i % 2 == 0 else coord * dh for i, coord in enumerate(segment)]
            yolo_segmentation.append(yolo_segment)
        return yolo_segmentation

    # Processing COCO annotations
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        # Create the corresponding YOLO label files for bboxes and masks
        bbox_filename = os.path.splitext(file_name)[0] + '.txt'
        bbox_filepath = os.path.join(bboxes_dir, bbox_filename)
        mask_filepath = os.path.join(masks_dir, bbox_filename)

        with open(bbox_filepath, 'w') as bbox_file, open(mask_filepath, 'w') as mask_file:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']

                    # Convert bounding boxes
                    if 'bbox' in annotation:
                        bbox = convert_bbox((width, height), annotation['bbox'])
                        bbox_file.write(f"{category_id} {' '.join(map(str, bbox))}\n")

                    # Convert segmentations
                    if 'segmentation' in annotation:
                        yolo_segmentation = convert_segmentation((width, height), annotation['segmentation'])
                        for seg in yolo_segmentation:
                            mask_file.write(f"{category_id} {' '.join(map(str, seg))}\n")

        # Copy the image to the output images directory
        src_image_path = os.path.join(images_dir, file_name)
        dst_image_path = os.path.join(output_images_dir, file_name)
        shutil.copy(src_image_path, dst_image_path)

    print("Conversion and copying of images completed.")

def create_yaml_file(output_folder, json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    categories = coco_data.get('categories', [])
    class_names = [category['name'] for category in categories]
    num_classes = len(class_names)
    data = {
        'train': os.path.join(output_folder, 'train', 'images'),
        'val': os.path.join(output_folder, 'val', 'images'),
        'test': os.path.join(output_folder, 'test', 'images'),
        'nc': num_classes,
        'names': class_names
    }

    yaml_path = os.path.join(output_folder, 'data.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"YAML file created: {yaml_path}")

if __name__ == "__main__":
    # Paths
    input_json_path = 'blue_truck_cap_simple_coco/train.json'
    output_dir = 'blue_truck_cap_dataset_yolo/train'  # Main directory for the output
    convert2yolo(input_json_path, output_dir)
    input_json_path = 'blue_truck_cap_simple_coco/validation.json'
    output_dir = 'blue_truck_cap_dataset_yolo/val'  # Main directory for the output
    convert2yolo(input_json_path, output_dir)
    input_json_path = 'blue_truck_cap_simple_coco/test.json'
    output_dir = 'blue_truck_cap_dataset_yolo/test'  # Main directory for the output
    convert2yolo(input_json_path, output_dir)
    output_folder = os.path.dirname(output_dir)
    create_yaml_file(output_folder, input_json_path)
