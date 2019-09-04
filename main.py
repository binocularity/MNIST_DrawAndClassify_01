#
# NSH August 2019
#
# Draw a number, save the drawing, classify the drawing with MNIST pretrained CNN
#
# Some background links that helped create this are below:
#
# https://www.codementor.io/kiok46/beginner-kivy-tutorial-basic-crash-course-for-apps-in-kivy-y2ubiq0gz
#
#
#

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.button import Label
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.core.window import Window
from kivy.graphics import Color, Line

#
# Tensorflow stuff.
#
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import plot_model
from keras.models import Model
from keras import activations
import keras
import keras.backend as K
from vis.utils import utils

#
# Utils
#
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os import listdir 
import matplotlib.cm as cm


#Disabling developer warning messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#
# Zero the image for the graph so it is display blank to start with.
# 
import shutil
shutil.copy("blank.png","classified.png")
shutil.copy("blank.png","cnnInput.png")
shutil.copy("blank.png","saliency.png")

# Dummy call to overcome problems with matplotlib changing the window size on first call
plt.figure(figsize=(10,10))

Builder.load_file("buttons.kv")

# Utility to plot saliency
def plot_map(grads, img):
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(img)
    axes[1].imshow(img)
    i = axes[1].imshow(grads,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    plt.suptitle("Saliency plot")
    plt.show()

class Container(BoxLayout):
    display = ObjectProperty()

class LabelDraw(Label):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 0)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y),width = 15)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def clear_canvas(self, obj):
        self.canvas.clear()

    def save_canvas(self,obj):
     self.export_as_image().save("userDrawn.png", flipped=False)   

class ClrButton(Button):
    pass

class SaveButton(Button):
    pass

class SaliencyButton(Button):
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        self.model = load_model('MNIST_model.h5')

        print("")
        print(">>>>>>>>>")
        print("Saliency model loaded")
        print("")
        

    def saliency_image(self):
        layer_outputs = [layer.output for layer in self.model.layers]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)

        Custom_image_dir = './'
        Custom_image = "userDrawn.png"
        img = image.load_img('%s/%s' %(Custom_image_dir, Custom_image),  target_size=(28, 28), color_mode="grayscale")
        img_arr = image.img_to_array(img)
        img_data = np.expand_dims(img_arr,0)
        #Rescale RGB values.
        img_data /= 255.0

        layer_idx = utils.find_layer_idx(self.model, 'dense_2')
        # Swap softmax with linear
        self.model.layers[layer_idx].activation = keras.activations.linear
        self.model = utils.apply_modifications(self.model)

        #Check we can still categorise the image.
        prediction =self.model.predict(img_data)
        prediction_category = np.argmax(prediction[0])
        print("Predicted numeral modified model: ", prediction_category)

        class_idxs_sorted = np.argsort(prediction.flatten())[::-1]
        class_idx = class_idxs_sorted[0]

        ## select class of interest
        class_idx         = class_idxs_sorted[0]
        ## define derivative d loss / d layer_input
        layer_input       = self.model.input
        ## This model must already use linear activation for the final layer
        loss              = self.model.layers[layer_idx].output[...,class_idx]
        grad_tensor       = K.gradients(loss,layer_input)[0]

        ## create function that evaluate the gradient for a given input
        # This function accept numpy array
        derivative_fn     = K.function([layer_input],[grad_tensor])

        ## evaluate the derivative_fn
        grad_eval_by_hand = derivative_fn([img_data[...]])[0]
        print(grad_eval_by_hand.shape)


        grad_eval_by_hand = np.abs(grad_eval_by_hand).max(axis=(0,3))

        ## normalize to range between 0 and 1
        arr_min, arr_max  = np.min(grad_eval_by_hand), np.max(grad_eval_by_hand)
        print("Min Max: ", arr_min, arr_max)
        grad_eval_by_hand = (grad_eval_by_hand - arr_min) / (arr_max - arr_min + K.epsilon())
        #grad_eval_by_hand = 1.0-(grad_eval_by_hand - arr_min) / (arr_max - arr_min + K.epsilon())

        plt.figure(figsize=(10,10))
        plt.title('Saliency Map')
        plt.imshow(grad_eval_by_hand,cmap="coolwarm",alpha=0.8)
        plt.savefig('saliency.png')
        plt.close("all")

class ClassifyButton(Button):
    
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        self.model = load_model('MNIST_model.h5')

        print("")
        print(">>>>>>>>>")
        print("Main model loaded")
        print("")
        
        # print("")
        # print(">>>>>>>>>")
        # print("Model summary")
        # print( self.model.summary() )
        # print("")
        # print("Model layers")
        # for layer in self.model.layers:
        #     print(layer.name, layer.input.shape, layer.output.shape)
        # print("<<<<<<<<<")          
        # print("")


    def classify_image(self):
        Custom_image_dir = './'
        Custom_image = "userDrawn.png"
        img = image.load_img('%s/%s' %(Custom_image_dir, Custom_image),  target_size=(28, 28), color_mode="grayscale")

        plt.figure(figsize=(10,10))
        plt.title('Actual input to CNN')
        plt.imshow(img,cmap=cm.gray, vmin=0, vmax=255)
        plt.savefig('cnnInput.png')
        plt.close("all")
        
        #Add batch number, required for model.
        img = image.img_to_array(img)
        img_data = np.expand_dims(img,0)
        #Rescale RGB values.
        img_data /= 255.0
        #Predict custom image.
        prediction = self.model.predict(img_data)
        print(prediction.argmax())

        plt.figure(figsize=(10,10))
        index = np.arange(10)
        my_cmap = cm.get_cmap('winter')
        plt.bar(index, prediction[0],color=my_cmap(prediction[0]))
        plt.title('Predicted: %s' %np.argmax(prediction[0]))
        plt.xticks(index)
        plt.yticks(np.arange(0, 1.1, step = 0.1))
        #plt.show(block=False)
        plt.savefig('classified.png')
        plt.close("all")
        
class MainApp(App):

    def build(self):
        self.title = 'Convolutional Neural Network, handwritten number classification'
        return Container()

if __name__ == "__main__":
    Window.size = (1440,960)
    app = MainApp()
    app.run()