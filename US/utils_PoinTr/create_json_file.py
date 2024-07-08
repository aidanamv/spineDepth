import json
import os
import shutil
from scipy.spatial import KDTree, ConvexHull

directory_path_partial = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/train/partial/10102023"
directory_path_complete = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset/val/partial/10102023"

train_files = os.listdir(directory_path_partial)
test_files = os.listdir(directory_path_complete)


data = {
    "taxonomy_id": "10102023",
    "taxonomy_name": "vertebrae",
    "train": train_files,
    "val": test_files

}

print(len(test_files))
print(len(train_files))


save_dir = "/Users/aidanamassalimova/Documents/US_Paper/Point_Tr_Dataset"
# Create a JSON file and write the data
with open(save_dir + '/fold_0.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)