import os
import shutil

def rename_and_copy_files(source_path, target_path, class_id):
    # Get the name of the dataset from the source path
    dataset_name = os.path.basename(os.path.normpath(source_path))
    
    # Iterate through the dataset subsets (train, test, val)
    for subset in ['train', 'test', 'val']:
        # Iterate through the different file types (images, bounding boxes, masks)
        for filetype in ['images', 'labels/bboxes', 'labels/masks']:
            # Define the source and target directories for each subset and file type
            source_dir = os.path.join(source_path, subset, filetype)
            target_dir = os.path.join(target_path, subset, filetype)
            
            # Create the target directory if it doesn't exist
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            # Iterate through all the files in the source directory
            for filename in os.listdir(source_dir):
                # Create a new filename by prefixing the dataset name
                new_filename = f"{dataset_name}_{filename}"
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, new_filename)
                
                # If the file is a label file (bbox or mask), update the class ID and save it
                if filetype in ['labels/bboxes', 'labels/masks'] and filename.endswith('.txt'):
                    update_class_id(source_file, target_file, class_id)
                else:
                    # Otherwise, simply copy the file to the target directory
                    shutil.copyfile(source_file, target_file)

def update_class_id(source_file, target_file, new_class_id):
    # Open the source file and read its lines
    with open(source_file, 'r') as f:
        lines = f.readlines()
    
    # Open the target file for writing with the new class ID
    with open(target_file, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            # Replace the class ID with the new one
            parts[0] = str(new_class_id)
            # Write the updated line to the target file
            f.write(' '.join(parts) + '\n')

def merge_datasets(dataset1_path, dataset2_path, merged_dataset_path):
    # Assign class IDs to each dataset to differentiate objects
    class_id_dataset1 = 0
    class_id_dataset2 = 1

    # Copy and rename files from dataset 1 with class ID 0
    rename_and_copy_files(dataset1_path, merged_dataset_path, class_id_dataset1)
    # Copy and rename files from dataset 2 with class ID 1
    rename_and_copy_files(dataset2_path, merged_dataset_path, class_id_dataset2)

# Example usage of the script
if __name__ == "__main__":
    # Define paths for dataset 1, dataset 2, and the merged dataset
    dataset1_path = 'red_truck_cab_dataset_yolo'
    dataset2_path = 'blue_truck_cab_dataset_yolo'
    merged_dataset_path = 'truck_cab_dataset_merged_1'

    # Call the merge_datasets function to merge both datasets
    merge_datasets(dataset1_path, dataset2_path, merged_dataset_path)

    print("Datasets merged successfully!")
