import random
import os
import math
from shutil import copyfile
from shutil import copytree
import shutil

seed = 1
random.seed(seed)

data_dir = './dataset_original/ZhangLabData/CellData'

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 1 - train_ratio - val_ratio

dest_folder_train = './CELL_data_split_by_person/train'
dest_folder_val = './CELL_data_split_by_person/val'
dest_folder_test = './CELL_data_split_by_person/test'

classes = os.listdir(data_dir)

def safe_create_dir(directory):
    try:
        os.makedirs(directory)
        print ("Successfully created the directory %s " % directory)
    except OSError:
        print ("Creation of the directory %s failed" % directory)

for subdir in classes:
    if os.path.exists(os.path.join(dest_folder_train,subdir)):
        shutil.rmtree(os.path.join(dest_folder_train,subdir))

    if os.path.exists(os.path.join(dest_folder_val,subdir)):
        shutil.rmtree(os.path.join(dest_folder_val,subdir))

    if os.path.exists(os.path.join(dest_folder_test,subdir)):
        shutil.rmtree(os.path.join(dest_folder_test,subdir))

    safe_create_dir(os.path.join(dest_folder_train, subdir))
    safe_create_dir(os.path.join(dest_folder_val, subdir))
    safe_create_dir(os.path.join(dest_folder_test, subdir))

    sub_data_dir = os.path.join(data_dir, subdir)
    person_count = len(os.listdir(sub_data_dir))
    
    train_count = math.ceil(train_ratio * person_count)
    val_count = math.ceil(val_ratio * person_count)

    person_list = os.listdir(sub_data_dir)
    random.shuffle(person_list)
    person_list_train = person_list[0:train_count]
    person_list_val = person_list[train_count:train_count+val_count]
    person_list_test = person_list[train_count+val_count:]
    
    train_counter, validation_counter, test_counter = 0, 0, 0
    for person in os.listdir(sub_data_dir):
        sub_sub_data_dir = os.path.join(sub_data_dir, person)
        if os.path.isdir(sub_sub_data_dir):  # Check if the path is a directory
            filelists = os.listdir(sub_sub_data_dir)
            for filename in filelists:
                if os.path.isfile(os.path.join(sub_sub_data_dir, filename)): 
                    if person in person_list_train:
                        dest_filename = subdir + str(train_counter)+'.jpg'
                        dest_folder_train_subdir = os.path.join(dest_folder_train,subdir)
                        copyfile(os.path.join(sub_sub_data_dir, filename), os.path.join(dest_folder_train_subdir, dest_filename))
                        train_counter += 1
                    elif person in person_list_val:
                        dest_filename = subdir + str(validation_counter)+'.jpg'
                        dest_folder_val_subdir = os.path.join(dest_folder_val, subdir)
                        copyfile(os.path.join(sub_sub_data_dir, filename), os.path.join(dest_folder_val_subdir, dest_filename))
                        validation_counter += 1
                    else:
                        dest_filename = subdir + str(test_counter)+'.jpg'
                        dest_folder_test_subdir = os.path.join(dest_folder_test, subdir)
                        copyfile(os.path.join(sub_sub_data_dir, filename), os.path.join(dest_folder_test_subdir, dest_filename))
                        test_counter += 1

    print("Copy {} files to train\{}".format(train_counter,subdir))
    print("Copy {} files to val\{}".format(validation_counter, subdir))
    print("Copy {} files to test\{}".format(test_counter, subdir))

print("End")