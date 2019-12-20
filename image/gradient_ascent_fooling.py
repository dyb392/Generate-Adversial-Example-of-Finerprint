"""
Created on Thu Oct 28 08:12:10 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np
import csv
from torch.optim import SGD
from torchvision import models
from torch.nn import functional
from myNet import CNNnet
from misc_functions import preprocess_image, recreate_image
import torch

class FoolingSampleGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent, breaks as soon as
        the target prediction confidence is captured
    """
    def __init__(self, model, target_class, minimum_confidence, img_path):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.minimum_confidence = minimum_confidence
        # Generate a random image
        self.created_image = cv2.imread(img_path, 1)

    def generate(self, saveFile):
        for i in range(1, 15):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=1)
            # Forward
            output = self.model(self.processed_image)
            #print(functional.softmax(output)[0][self.target_class].data.numpy())
            # Get confidence from softmax
            target_confidence = functional.softmax(output)[0][self.target_class].data.numpy()
            if target_confidence > self.minimum_confidence and i != 1:
                # Reading the raw image and pushing it through model to see the prediction
                # this is needed because the format of preprocessed image is float and when
                # it is written back to file it is converted to uint8, so there is a chance that
                # there are some losses while writing
                confirmation_image = cv2.imread(saveFile, 1)
                # Preprocess image
                confirmation_processed_image = preprocess_image(confirmation_image)
                # Get prediction
                confirmation_output = self.model(confirmation_processed_image)
                # Get confidence
                softmax_confirmation = \
                    functional.softmax(confirmation_output)[0][self.target_class].data.numpy()
                if softmax_confirmation > self.minimum_confidence:
                    print('Generated fooling image with', "{0:.2f}".format(softmax_confirmation),
                          'confidence at', str(i) + 'th iteration.')
                    break
            # Target specific class
            class_loss = -output[0, self.target_class]
            print('Iteration:', str(i), 'Target Confidence', "{0:.4f}".format(target_confidence))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            # Save image
            cv2.imwrite(saveFile, self.created_image)
        return self.processed_image


if __name__ == '__main__':
    test = []
    with open("test.csv", "r", newline="") as r:
        reader = csv.reader(r)
        line = 0
        for row in reader:
            if line == 1:
                test.append(row)
            line = 1

    for image in range(len(test)):
        # Define model
        pretrained_model = CNNnet()
        #print(pretrained_model)
        pretrained_model.load_state_dict(torch.load(r"./model"))
        #image = 0
        if image > len(test)-250:
            target_image = image-200
        else:
            target_image = image+200
        filename = test[image][0].split(".")
        filenames = filename[0].split("/")
        processed = "FOOLgenerated/"+filenames[1]+"/"+filenames[len(filenames)-1]+"_FOOL.jpeg"
        target_class = int(test[target_image][1])
        print(test[image][1], target_class)
        cig = FoolingSampleGeneration(pretrained_model, target_class, 0.7, test[image][0])
        cig.generate(processed)
        print(image)
