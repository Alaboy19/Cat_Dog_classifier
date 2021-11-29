import os
import glob
import cv2
import tqdm
import pandas as pd
import shutil

filenames = list(glob.iglob('data/train/*.jpg', recursive=True))

print(len(filenames))

with open('csv/train_dataframe.csv', 'w') as csv:
    csv.write('filename, label\n')  # 0 for cat and 1 for dog
    hashLp = {}
    # hashData = {}
    for filename in filenames:
        img = cv2.imread(filename)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if var < 200 or img is None:
            continue

        label = 0 if 'cat' in filename else 1

        csv.write(f'{filename},{label}\n')

df = pd.read_csv('csv/train_dataframe.csv')
print(df.head)
print(df.shape)
