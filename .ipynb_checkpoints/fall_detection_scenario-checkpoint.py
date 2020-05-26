import tensorflow as tf
import pandas as pd
from PIL import Image
import io
import numpy as np
import random
import sys
from fd_module import FallDetection

train1= pd.read_csv("body_posture_dataset/train_1.csv")
train2= pd.read_csv("body_posture_dataset/train_2.csv")
train3= pd.read_csv("body_posture_dataset/train_3.csv")


filename1 = train1["filename"].tolist()
xmin1 = train1["xmin"].tolist()
ymin1 = train1["ymin"].tolist()
xmax1 = train1["xmax"].tolist()
ymax1 = train1["ymax"].tolist()

filename2 = train2["filename"].tolist()
xmin2 = train2["xmin"].tolist()
ymin2 = train2["ymin"].tolist()
xmax2 = train2["xmax"].tolist()
ymax2 = train2["ymax"].tolist()

filename3 = train3["filename"].tolist()
xmin3 = train3["xmin"].tolist()
ymin3 = train3["ymin"].tolist()
xmax3 = train3["xmax"].tolist()
ymax3 = train3["ymax"].tolist()

stream = []
for i, _ in enumerate(filename1):
    temp = [filename1[i], [xmin1[i], ymin1[i], xmax1[i], ymax1[i]], 0] 
    stream.append(temp)

for i, _ in enumerate(filename2):
    temp = [filename2[i], [xmin2[i], ymin2[i], xmax2[i], ymax2[i]], 2] 
    stream.append(temp)

for i, _ in enumerate(filename3):
    temp = [filename3[i], [xmin3[i], ymin3[i], xmax3[i], ymax3[i]], 1] 
    stream.append(temp)
    
#stream = np.asarray(stream)
#random.seed(0)
random.shuffle(stream)


# inference

fd = FallDetection(20, 5, 7) #buffer size, fall_threshold, long_lie_threshold




for item in stream:
    image = item[0]
    bbox = item[1]
    label = item[2]
    image = fd.get_filepath(image, label)
    image = fd.preprocess_image(image, bbox)

    # inference
    prediction = fd.inference(image)

    # fall detection
    fd.buffer_step(prediction)
    if fd.detect_fall() and fd.detect_long_lie():
        fd.generate_alarm()
