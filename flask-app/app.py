from flask import Flask, request
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import parse_json, iterate_over_json, refactor_weights
from model import Cat_Dog
import argparse
import timm
import torch
import json

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model to run', default='custom')
    args = parser.parse_args()

    if args.model == 'custom':
        model = Cat_Dog()
        weights = torch.load(
            '../weights/CatDog_epoch=188_train_loss=0.2179_val_loss=0.2807.ckpt',
            map_location='cpu')[
            'state_dict']
        weights = refactor_weights(weights)
    else:
        model = timm.create_model('efficientnet_b0', num_classes=2)
        weights = torch.load(
            '../weights/CatDog_epoch=61_val_loss=0.0585_val_acc=0.9848.ckpt',
            map_location='cpu')[
            'state_dict']
        weights = refactor_weights(weights)

    model.load_state_dict(weights)
    model.eval()

    image_list = parse_json(request.json)
    jsn_out = iterate_over_json(image_list, model)

    return json.dumps(jsn_out)


if __name__ == "__main__":
    app.run(debug=True)
