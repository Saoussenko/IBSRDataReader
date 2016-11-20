import os

import numpy as np

import settings
from utils import perform_rotation
from read_data import get_file


my_order_file_address = '/home/siavash/programming/thesis/meta/my_order.txt'
label_directory = '/home/siavash/programming/thesis/labels/'
nifti_file_base = '/home/siavash/programming/thesis/nifti_files/'
np_file_addresses = '/home/siavash/programming/thesis/npy_files/'
pre_processed_nifti_folder = '/home/siavash/programming/thesis/preprocessed_nifti_files/'
pre_processed_numpy_folder = '/home/siavash/programming/thesis/preprocessed_numpy_files/'


def create_npy():
    counter = 0

    headers = [x for x in os.listdir(settings.LABELS_ADDRESS) if x.endswith(settings.HEADER_FILES_SUFFIX)]
    for file_name in headers:
        header_file_address = label_directory + file_name
        with open(header_file_address) as header_file:
            number_of_cuts = int(header_file.readline().split()[2])
            print header_file_address
            print number_of_cuts
            image_3d = np.zeros([256, number_of_cuts, 256], dtype='>u2')
            lbl_3d = np.zeros([256, number_of_cuts, 256], dtype='>u2')
            for i in range(number_of_cuts):
                img, lbl = get_file(counter)
                img = perform_rotation(img)
                lbl = perform_rotation(lbl)

                image_3d[:, i, :] = img
                lbl_3d[:, i, :] = lbl
                counter += 1

            np.save(np_file_addresses + file_name[:-4] + '.npy', image_3d)
            np.save(np_file_addresses + file_name[:-4] + '_lbl.npy', lbl_3d)


def create_nii_from_binary():
    import nifti
    np_files = [f for f in os.listdir(np_file_addresses) if f.endswith('.npy')]
    for np_file in np_files:
        image = np.load(np_file_addresses + np_file)
        print image.shape
        nifti_image = nifti.NiftiImage(image.astype('uint16'))
        nifti_image.save(nifti_file_base + np_file[:-4] + '.nii')


def convert_nifti_to_numpy(nifti_file_address):
    import nifti
    return nifti.NiftiImage(nifti_file_address).getDataArray()


def convert_nifti_folder_to_numpy(nifti_folder, numpy_folder):
    for file_address in os.listdir(nifti_folder):
        numpy_array = convert_nifti_to_numpy(nifti_folder + file_address)
        np.save(numpy_folder + file_address[:-4] + '.npy', numpy_array)


# First Run This
create_npy()
create_nii_from_binary()

# Run Bash Script and then the following line
convert_nifti_folder_to_numpy(pre_processed_nifti_folder, pre_processed_numpy_folder)
