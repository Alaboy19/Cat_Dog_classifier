import cv2
import json
import config
import torch
import numpy as np
import base64
from torch.nn.functional import softmax as sft


def parse_json(json_input):  # return the byte_string
    return json_input["photos"]  # list of dicts


def read_img(img_paths):
    imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        imgs.append(img)
    return imgs


def convert_to_base64(img):
    retval, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def convert_to_np(byte_string):
    jpg_original = base64.b64decode(byte_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img_np = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    return img_np


def convert_to_json(list_of_5img):
    output = {"photos": []}
    counter = 0
    for img in list_of_5img:
        img_dict = {}
        counter += 1
        # img_dict['ID']
        img_dict['ID'] = f'img{counter}'
        img_dict['img_code'] = f'{convert_to_base64(img)}'
        output['photos'].append(img_dict)
    with open('test_input.json', 'w') as f:
        json.dump(output, f)


def image_preprocess(image_in):
    image_in = cv2.resize(image_in, (config.image_w, config.image_h))
    image_in = image_in.astype(np.float32) / 255.
    image_in = np.transpose(image_in, (2, 0, 1))
    image_in = np.expand_dims(image_in, 0)
    image_in = torch.tensor(image_in)
    return image_in


def iterate_over_json(photos, model):  # input from parse json
    results = {"results": []}
    for photo in photos:
        result_dict = {}
        ID, img_code = photo['ID'], photo['img_code']
        img_np = convert_to_np(img_code)  # returns none type
        image = image_preprocess(img_np)
        with torch.no_grad():
            output = model(image)
            cat_prob, dog_prob = sft(output, dim=1)[0].tolist()
            cat_prob = round(cat_prob, 6)
            dog_prob = round(dog_prob, 6)
        result_dict["ID"] = ID
        result_dict['cat_prob'] = cat_prob
        result_dict['dog_prob'] = dog_prob
        results["results"].append(result_dict)

    return results


def refactor_weights(weights):
    new_weights = {}

    for k, v in weights.items():
        new_weights[k.replace('module.', '').replace('model.', '')] = v
    return new_weights


if __name__ == "__main__":
    image = 'json_sample/1.jpg'
    image = cv2.imread(image)
    img_np = convert_to_np(convert_to_base64(image))
