import json
from collections import OrderedDict
import glob
import shutil
import os
import numpy as np
img_list = glob.glob('./eg3d/projector_test_data/*.png')
with open('E:/data/FFHQ/ed3g/dataset.json', 'rb') as f:
    json_data = json.load(f, object_pairs_hook=OrderedDict)
labels = json_data["labels"]

print(img_list)
for i,img_path in enumerate(img_list):
    image_name = os.path.basename(img_path)[:-4]
    id = int(image_name)*2
    label = labels[id][1]
    print(image_name,labels[id])
    np.save(f'./eg3d/projector_test_data/{image_name}.npy',np.array(label))