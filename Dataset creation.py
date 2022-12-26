import os
from random import randrange
from random import seed
import shutil

# # UNZIP THE FILES
# import zipfile
# OGData_path = r'C:\Users\janal\Downloads\DataAI project\Swedish Leaves'
# zip_files = os.listdir(OGData_path)
# print(zip_files)
#
# for zip_file in zip_files:
#     local_zip = os.path.join(r'C:\Users\janal\Downloads\DataAI project\Swedish Leaves', zip_file)
#     zip_ref = zipfile.ZipFile(local_zip, 'r')
#     zip_ref.extractall(r'C:\Users\janal\Downloads\DataAI project\Extracted' + zip_file)
#     zip_ref.close()
#
# # Erase existing Train_set and Validation_set directories
# # COMMENT this if you do not want to create new dataset partitions.
#
# try:
#     shutil.rmtree(r'C:\Users\janal\Downloads\DataAI project\Train_set')
#     shutil.rmtree(r'C:\Users\janal\Downloads\DataAI project\Validation_set')
#     shutil.rmtree(r'C:\Users\janal\Downloads\DataAI project\Test_set')
# except OSError:
#     pass
#
# #Create directories for train, validation and test sets
# try:
#     os.mkdir(r'C:\Users\janal\Downloads\DataAI project\Train_set')
#     os.mkdir(r'C:\Users\janal\Downloads\DataAI project\Validation_set')
#     os.mkdir(r'C:\Users\janal\Downloads\DataAI project\Test_set')
# except OSError:
#     pass

#Set up directory paths for train, validation, and test sets

train_set_dir = r'C:\Users\janal\Downloads\DataAI project\Train_set'
validation_set_dir = r'C:\Users\janal\Downloads\DataAI project\Validation_set'
test_set_dir = r'C:\Users\janal\Downloads\DataAI project\Test_set'

COMPLETE_DATA_PATH = r'C:\Users\janal\Downloads\DataAI project\Extracted'
Datafolders = os.listdir(COMPLETE_DATA_PATH)
print(Datafolders)

for Folder in Datafolders:
    Train_set_NewFolder = os.path.join(train_set_dir, Folder)
    Validation_set_NewFolder = os.path.join(validation_set_dir, Folder)
    Test_set_NewFolder = os.path.join(test_set_dir, Folder)
    try:
        os.mkdir(Train_set_NewFolder)
        os.mkdir(Validation_set_NewFolder)
        os.mkdir(Test_set_NewFolder)
    except OSError:
        pass
    current_path = os.path.join(COMPLETE_DATA_PATH, Folder)
    print(current_path)
    os.chdir(current_path)
    # print(os.listdir())
    file_names = os.listdir()
    training_set_size = int(0.75*len(os.listdir()))
    # print(training_set_size)
    validation_set_size = int(0.20*len(os.listdir()))                                       #DETERMINES THE LENGTH OF THE VALIDATION SET BASED ON THE NUMBER OF EXAMPLES OF THE WHOLE DATASET
    # print(validation_set_size)
    test_set_size = len(os.listdir()) - training_set_size - validation_set_size             #DETERMINES THE LENGTH OF THE TEST SET BASED ON LENGTHS OF THE TEST AND VALIDATION SET, AND THE WHOLE DATASET

    img_path = [os.path.join(current_path, img_name) for img_name in os.listdir()]
    # print(img_path[0])
    img_path_copy = img_path


    while(len(os.listdir(Train_set_NewFolder))<=training_set_size-1):
        print(len(os.listdir(train_set_dir)))
        index = randrange(len(img_path_copy))
        shutil.copyfile(img_path_copy[index], os.path.join(Train_set_NewFolder, file_names[index]))
        img_path_copy.pop(index)
        file_names.pop(index)
    while(len(os.listdir(Validation_set_NewFolder))<=validation_set_size-1):
        index = randrange(len(img_path_copy))
        shutil.copyfile(img_path_copy[index], os.path.join(Validation_set_NewFolder, file_names[index]))
        img_path_copy.pop(index)
        file_names.pop(index)

    for img in range(len(img_path_copy)):
        shutil.copyfile(img_path_copy[img], os.path.join(Test_set_NewFolder, file_names[img]))