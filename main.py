from utils import parse_json, iterate_over_json, refactor_weights
import model
import argparse
import timm
import torch
import cv2
from torch.nn.functional import softmax as sft


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model to run', default='custom')
args = parser.parse_args()

if args.model == 'custom':
    model = model.Cat_Dog()
    weights = torch.load(
        'weights/CatDog_epoch=188_train_loss=0.2179_val_loss=0.2807.ckpt')[
        'state_dict']
    weights = refactor_weights(weights)
else:
    model = timm.create_model('efficientnet_b0', num_classes=2)
    weights = torch.load(
        'weights/CatDog_epoch=61_val_loss=0.0585_val_acc=0.9848.ckpt')[
        'state_dict']
    weights = refactor_weights(weights)

model.load_state_dict(weights)
model.eval()

input_json = 'json_sample/test_input.json'
with open(input_json, 'r') as j:
    contents = j.read()

image_list = parse_json(contents)
print(len(image_list))
jsn_out = iterate_over_json(image_list, model)
with open('json_sample/test_output.json', 'w') as j:
    j.write(jsn_out)
