import json
from collections import OrderedDict
import glob
import shutil
import os
import numpy as np
img_list = glob.glob('E:/data/FFHQ/imgs1024/00000/*')
with open('E:/data/FFHQ/ed3g/dataset.json', 'rb') as f:
    json_data = json.load(f, object_pairs_hook=OrderedDict)
labels = json_data["labels"]
for i,img_path in enumerate(img_list[:10]):
    image_name = os.path.basename(img_path)[:-4]
    shutil.copy(img_path,f'./projector_test_data/{image_name}.png')


    label = labels[i][1]

    np.save(f'./projector_test_data/{image_name}.npy',np.array(label))