import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data/Stanford3dDataset_v1.2_Aligned_Version')
import indoor3d_util

anno_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'meta/anno_paths.txt'))]
anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]

output_folder = os.path.join(ROOT_DIR, 'data/stanford_indoor3d') 
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

revise_file = os.path.join(DATA_PATH, "Area_5/hallway_6/Annotations/ceiling_1.txt")
with open(revise_file, "r") as f:
    data = f.read()
    data = data[:5545347] + ' ' + data[5545348:]
    f.close()
with open(revise_file, "w") as f:
    f.write(data)
    f.close()

for anno_path in anno_paths:
	print(anno_path)
	elements = anno_path.split('/')
	out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
	indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')