import sys
import numpy as np
from PIL import Image
import tensorflow as tf


class FallDetection:
    def __init__(self, buffer_size, fall_threshold, long_lie_threshold):

        
        # MODEL_PATH='model/basic-cnn-functional.h5'
        # MODEL_PATH='model/basic-cnn-functional-expert.h5'
        # MODEL_PATH='model/basic-cnn-sequential.h5'
        self.MODEL_PATH = 'model/basic-cnn-sequential-expert.h5'
        self.model = tf.keras.models.load_model(self.MODEL_PATH)
        # inference를 할 경우엔 warning 무시
        # training을 할 경우엔 compile 해야함

        #----
        
        
        self.buffer_size = buffer_size
        self.fall_threshold = fall_threshold
        self.long_lie_threshold = long_lie_threshold

        self.buffer = []
        self.long_lie_window = []
        self.lying_cnt = 0

        self.STANDING = 0
        self.LYING = 1
        self.BENDING = 2

        
    def inference(self, image):
        prediction = self.model(image)
        #self.model.predict(image)
        return np.argmax(prediction)

    
    def get_filepath(self, filename, label):
        return "body_posture_dataset/" + str(label) + "/" + filename

    
    def preprocess_image(self, image, bbox):
        img = Image.open(image).convert("L")  # convert to grayscale
        img = img.crop(bbox)
        img = img.resize((50, 50))
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, -1)
        img = np.expand_dims(img, 0)

        return img
        
        
    def buffer_step(self, label):
        # self.lying_cnt = 0
        self.buffer.append(label)

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def detect_fall(self):
        # st: standing timestamp
        # lt: lying timestamp
        for st, label in enumerate(self.buffer):
            if label == self.STANDING:
                for lt in range(st, st + self.fall_threshold):
                    if lt > len(self.buffer) - 1:
                        break
                    if self.buffer[lt] == self.LYING:
                        self.st = st
                        self.lt = lt
                        return True

        return False

    def detect_long_lie(self):

        self.lying_cnt = 0
        self.long_lie_window = []

        # 1. declaring sliding window
        for t in range(self.lt, self.lt + self.long_lie_threshold):
            if t > len(self.buffer) - 1:
                break

            self.long_lie_window.append(self.buffer[t])

            # initiate lying count
            if self.buffer[t] == self.LYING:
                self.lying_cnt += 1

        # alarm condition
        if self.lying_cnt >= self.long_lie_threshold:
            return True

        else:
            return False

    def generate_alarm(self):
        sys.stdout.write("[ALERT] fall-down has just occurred!\n")
        sys.stdout.write("fall detected between " + str(self.st) + " and " + str(self.lt)+"\n")
        sys.stdout.write("self.buffer: {} \n".format(self.buffer))
        sys.stdout.write("--------\n")

