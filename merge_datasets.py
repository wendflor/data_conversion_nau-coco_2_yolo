import os
import shutil

def rename_and_copy_files(source_path, target_path, class_id):
    dataset_name = os.path.basename(os.path.normpath(source_path))
    
    for subset in ['train', 'test', 'val']:
        for filetype in ['images', 'labels/bboxes', 'labels/masks']:
            source_dir = os.path.join(source_path, subset, filetype)
            target_dir = os.path.join(target_path, subset, filetype)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            for filename in os.listdir(source_dir):
                new_filename = f"{dataset_name}_{filename}"
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, new_filename)
                
                if filetype in ['labels/bboxes', 'labels/masks'] and filename.endswith('.txt'):
                    update_class_id(source_file, target_file, class_id)
                else:
                    shutil.copyfile(source_file, target_file)

def update_class_id(source_file, target_file, new_class_id):
    with open(source_file, 'r') as f:
        lines = f.readlines()
    
    with open(target_file, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            parts[0] = str(new_class_id)
            f.write(' '.join(parts) + '\n')

def merge_datasets(dataset1_path, dataset2_path, merged_dataset_path):
    # Assign class IDs to each dataset
    class_id_dataset1 = 0
    class_id_dataset2 = 1

    rename_and_copy_files(dataset1_path, merged_dataset_path, class_id_dataset1)
    rename_and_copy_files(dataset2_path, merged_dataset_path, class_id_dataset2)

# Example paths
if __name__ == "__main__":
    dataset1_path = 'red_truck_cap_dataset_yolo'
    dataset2_path = 'blue_truck_cap_dataset_yolo'
    merged_dataset_path = 'truck_cap_dataset_merged'

    merge_datasets(dataset1_path, dataset2_path, merged_dataset_path)

    print("Datasets merged successfully!")

