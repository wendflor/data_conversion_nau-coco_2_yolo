import os
import json
from shutil import copyfile

def convert_data(input_json_path, output_folder):
    # Erstelle das Ausgabe-Verzeichnis für den Datensatz
    os.makedirs(output_folder, exist_ok=True)
    # Extrahiere den Basename der JSON-Datei ohne Endung
    json_filename = os.path.basename(input_json_path)
    json_basename = os.path.splitext(os.path.basename(input_json_path))[0]
    image_output_folder = os.path.join(output_folder, json_basename)

    dataset_name = os.path.basename(os.path.dirname(input_json_path))
    print(dataset_name)

    # Erstelle das Ausgabe-Verzeichnis für die Bilder
    os.makedirs(image_output_folder, exist_ok=True)

    # Load COCO json
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Map for storing image_id to filename
    # image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}

    # Rename and move images, update JSON file paths
    for image in coco_data['images']:
        old_filepath = os.path.join(dataset_name, image['file_name']).replace('\\', '/')
        new_filename = f"{str(image['id']).zfill(5)}.jpg"
        new_filepath = os.path.join(image_output_folder, new_filename).replace('\\', '/')
        
        # Copy and rename image files
        copyfile(old_filepath, new_filepath)
        image['file_name'] = new_filename  # Update file_name in COCO data

 
    # Extrahiere den Dateinamen

    print(f"Dateiname: {json_filename}")
    # Save the updated COCO json with new file paths
    updated_json_path = os.path.join(output_folder, json_filename).replace('\\', '/')
    os.makedirs(os.path.dirname(updated_json_path), exist_ok=True)
    with open(updated_json_path, 'w') as f:
        json.dump(coco_data, f)

    print("Bilder wurden umbenannt und verschoben, JSON-Datei wurde aktualisiert.")


if __name__ == "__main__":
    # Pfade anpassen
    input_json_path = 'truck_cab_dataset_red/train.json' # 
    output_folder = 'red_truck_cab_simple_coco/'  # dataset output folder
    convert_data(input_json_path, output_folder)
    input_json_path = 'truck_cab_dataset_red/validation.json'
    convert_data(input_json_path, output_folder)
    input_json_path = 'truck_cab_dataset_red/test.json'
    convert_data(input_json_path, output_folder)
