"""
Created on Fri Dec 15 19:57:34 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import csv
import os
import numpy as np
import cv2
import torch
from torch import nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None

from misc_functions import preprocess_image, recreate_image, get_params


class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('FGSMgenerated'):
            os.makedirs('FGSMgenerated')

    def generate(self, original_image, im_label, filename):
        # I honestly dont know a better way to create a variable with specific value
        im_label_as_var = Variable(torch.from_numpy(np.asarray([int(im_label)])))
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(10):
            print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            pred_loss = ce_loss(out, im_label_as_var.long())
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add Noise to processed image
            processed_image.data = processed_image.data + adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            confirmation_out = self.model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1)
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            # Check if the prediction is different than the original
            #if confirmation_prediction != im_label:
            '''
                print('Original image was predicted as:', im_label,
                      'with adversarial noise converted to:', confirmation_prediction,
                      'and predicted with confidence of:', confirmation_confidence)
                
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                cv2.imwrite(filename, noise_image)
            '''
        # Write image
        cv2.imwrite(filename, recreated_image)

        return 1


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
        target_example = test[image]  
        (original_image, prep_img, target_class, _, pretrained_model) =\
            get_params(target_example)
        filename = test[image][0].split(".")
        filenames = filename[0].split("/")
        processed = "FGSMgenerated/"+filenames[1]+"/"+filenames[len(filenames)-1]+"_FGSM.jpeg"
        FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.01)
        FGS_untargeted.generate(original_image, target_class, processed)
        print(image)
