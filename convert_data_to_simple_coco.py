import os
import json
from shutil import copyfile

def convert_data(input_json_path, output_folder):
    # Create the output directory for the dataset
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract the base name of the JSON file (without extension)
    json_filename = os.path.basename(input_json_path)
    json_basename = os.path.splitext(os.path.basename(input_json_path))[0]
    
    # Define the output folder for images
    image_output_folder = os.path.join(output_folder, json_basename)

    # Get the name of the dataset (parent folder of the input JSON file)
    dataset_name = os.path.basename(os.path.dirname(input_json_path))
    print(dataset_name)

    # Create the output directory for images
    os.makedirs(image_output_folder, exist_ok=True)

    # Load the COCO dataset from the input JSON file
    with open(input_json_path, 'r') as f:
        coco_data = json.load(f)

    # Prepare a map to store image_id to filename 
    # image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}

    # Loop over the images in the COCO data, rename, and move them
    for image in coco_data['images']:
        # Define the old file path for the image (based on the original dataset structure)
        old_filepath = os.path.join(dataset_name, image['file_name']).replace('\\', '/')
        
        # Create a new filename for the image using the image ID, padded with leading zeros
        new_filename = f"{str(image['id']).zfill(5)}.jpg"
        
        # Define the new file path in the output folder
        new_filepath = os.path.join(image_output_folder, new_filename).replace('\\', '/')
        
        # Copy the image from the old location to the new one, renaming it
        copyfile(old_filepath, new_filepath)
        
        # Update the 'file_name' field in the COCO JSON data to reflect the new filename
        image['file_name'] = new_filename

    # Print the extracted JSON filename
    print(f"Filename: {json_filename}")
    
    # Save the updated COCO JSON with the new file paths
    updated_json_path = os.path.join(output_folder, json_filename).replace('\\', '/')
    
    # Create the directory if it doesn't exist (just a precaution)
    os.makedirs(os.path.dirname(updated_json_path), exist_ok=True)
    
    # Write the updated COCO data to a new JSON file
    with open(updated_json_path, 'w') as f:
        json.dump(coco_data, f)

    print("Images have been renamed and moved, and the JSON file has been updated.")


if __name__ == "__main__":
    # Define the paths for the input JSON files and output folder
    input_json_path = 'truck_cab_dataset_red/train.json'  # Path to the train dataset
    output_folder = 'red_truck_cab_simple_coco/'  # Output folder for the converted dataset
    
    # Convert and update the train dataset
    convert_data(input_json_path, output_folder)
    
    # Convert and update the validation dataset
    input_json_path = 'truck_cab_dataset_red/validation.json'
    convert_data(input_json_path, output_folder)
    
    # Convert and update the test dataset
    input_json_path = 'truck_cab_dataset_red/test.json'
    convert_data(input_json_path, output_folder)
