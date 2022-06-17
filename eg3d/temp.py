import json
from collections import OrderedDict
import glob
import shutil
import os
import numpy as np

from  moviepy.editor import *
clipVideo =(VideoFileClip(r"D:\SoftApp\Python\MP4ToGIF\\MP4Data\3.mp4").subclip(0,16).resize(0.6))
clipVideo.write_gif(r"D:\SoftApp\Python\MP4ToGIF\MP4Data\3.gif", fps=2)
