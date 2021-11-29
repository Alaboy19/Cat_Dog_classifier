import json
import cv2
import numpy as np
import base64
from utils import read_img, convert_to_json

if __name__ == "__main__":
    img_paths = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    imgs = read_img(img_paths)
    jsn_out = convert_to_json(imgs)
    # lst_dicts = parse_json(jsn_out)
    # print(lst_dicts[0].keys())
