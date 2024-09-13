import json
import os
import shutil
import yaml

def convert2yolo(input_json_path, output_dir):

    # Extrahiere den Namen der JSON-Datei ohne die Endung
    json_basename = os.path.splitext(os.path.basename(input_json_path))[0]
    images_dir = os.path.join(os.path.dirname(input_json_path), json_basename) # source-image directory is expected to have the same name as .json-file

    output_images_dir = os.path.join(output_dir, 'images')  # Verzeichnis, in das die Bilder kopiert werden
    labels_dir = os.path.join(output_dir, 'labels')  # Verzeichnis, in das die Labels gespeichert werden

    # Erstelle die Ausgabe-Verzeichnisse, falls sie nicht existieren
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Lade COCO Annotationen
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Funktion zum Konvertieren der Bounding Box ins YOLO-Format
    def convert_bbox(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2] / 2.0) * dw
        y = (box[1] + box[3] / 2.0) * dh
        w = box[2] * dw
        h = box[3] * dh
        return (x, y, w, h)

    # Funktion zum Konvertieren der Segmentierung ins YOLO-Format
    def convert_segmentation(size, segmentation):
        dw = 1. / size[0]
        dh = 1. / size[1]
        yolo_segmentation = []
        for segment in segmentation:
            yolo_segment = [coord * dw if i % 2 == 0 else coord * dh for i, coord in enumerate(segment)]
            yolo_segmentation.append(yolo_segment)
        return yolo_segmentation

    # Verarbeitung der COCO Annotationen
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        # Erstelle das entsprechende YOLO-Label-Datei
        yolo_filename = os.path.splitext(file_name)[0] + '.txt'
        yolo_filepath = os.path.join(labels_dir, yolo_filename)

        with open(yolo_filepath, 'w') as yolo_file:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    category_id = annotation['category_id']
                    bbox = annotation['bbox']
                    segmentation = annotation['segmentation']
                    yolo_bbox = convert_bbox((width, height), bbox)
                    yolo_segmentation = convert_segmentation((width, height), segmentation)

                    # Sicherstellen, dass keine None-Werte geschrieben werden
                    if None not in yolo_bbox and category_id is not None:
                        # YOLO-Format: class_id x_center y_center width height
                        bbox_str = f"{category_id} {' '.join(map(str, yolo_bbox))}"
                        # Segmentierung hinzufügen: jede Koordinate der Segmentierung
                        for segment in yolo_segmentation:
                            seg_str = ' '.join(map(str, segment))
                            bbox_str += f" {seg_str}"
                        bbox_str += "\n"
                        yolo_file.write(bbox_str)

        # Kopiere das Bild in das train-Verzeichnis
        src_image_path = os.path.join(images_dir, file_name)
        dst_image_path = os.path.join(output_images_dir, file_name)
        shutil.copy(src_image_path, dst_image_path)

    print("Konvertierung und Kopieren der Bilder abgeschlossen.")

def create_yaml_file(output_folder, json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    categories = coco_data.get('categories', []) # Achtung in json.file auch dirstractor enthalten. Ergebnis falsch! Noch bearbeiten
    print("Achtung in json.file sind noch distractor Namen enthalten: Ergebnis/Einträge in YAML File potentiell falsch!")
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

    print(f"YAML-Datei wurde erstellt: {yaml_path}")


if __name__ == "__main__":
    # Pfade
    input_json_path = 'truck_cap_simple_coco/train.json'
    output_dir = 'truck_cap_dataset_yolo/train'     # Hauptverzeichnis für die Ausgabe
    convert2yolo(input_json_path, output_dir)
    input_json_path = 'truck_cap_simple_coco/validation.json'
    output_dir = 'truck_cap_dataset_yolo/val'     # Hauptverzeichnis für die Ausgabe
    convert2yolo(input_json_path, output_dir)
    input_json_path = 'truck_cap_simple_coco/test.json'
    output_dir = 'truck_cap_dataset_yolo/test'     # Hauptverzeichnis für die Ausgabe
    convert2yolo(input_json_path, output_dir)
    output_folder = os.path.dirname(output_dir)
    create_yaml_file(output_folder, input_json_path)