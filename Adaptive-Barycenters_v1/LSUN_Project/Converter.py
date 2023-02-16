import os
import numpy as np
from zipfile import ZipFile
import glob
import matplotlib.image as mpimg
from PIL import Image
import tensorflow as tf
from Load_Dataset import list_categories, download
from Read_Write import _export_mdb_images

'''Splits LSUN Bedrooms dataset into distinct subfolders according to their classes.'''


def LSUN_split():
    num = 0
    out_direc = '/home/mdedeogl/Desktop/LSUN_Exp/LSUN_dataset'
    if not os.path.exists(out_direc):
        os.mkdir(out_direc)
    listOfCats = list_categories()

    ind = 0
    for category in listOfCats:
        if not category == 'test':
            out_direc_sub = out_direc + '/' + category
            if not os.path.exists(out_direc_sub):
                os.mkdir(out_direc_sub)
                set_name = 'train'
                download(out_direc_sub, category, set_name)

            db_path = out_direc_sub + '/' + category + '_train_lmdb'
            final_dest = '/home/mdedeogl/Desktop/LSUN_Exp/LSUN_dataset/Class_' + str(ind)  # + '/Train'
            if not os.path.exists(final_dest):
                os.mkdir(final_dest)
            final_dest = final_dest + '/Train'
            if not os.path.exists(final_dest):
                os.mkdir(final_dest)
                fileName = db_path + '.zip'
                if not os.path.exists(db_path):
                    with ZipFile(fileName, 'r') as zipObj:
                        # Extract all the contents of zip file in current directory
                        zipObj.extractall(path=out_direc_sub)
                _export_mdb_images(db_path, ind, out_dir=final_dest, flat=True, limit=100000, size=64)
                os.chmod(db_path, 0o777)
                os.remove(db_path+'/data.mdb')
                os.remove(db_path + '/lock.mdb')
            ind += 1


if __name__ == '__main__':
    LSUN_split()
