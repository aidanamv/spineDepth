import os
import shutil

path = "/Users/aidanamassalimova/Documents/planning data original/stls all vertebrae"


files = os.listdir(path)

for file in files:

    specimen = file[:-7]
    print(specimen)

    new_dir = os.path.join(path, specimen)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    shutil.move(os.path.join(path,file), os.path.join(new_dir,file))

