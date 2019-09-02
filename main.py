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
from keras.utils import plot_model

from os import listdir 

#
# Tensorflow stuff.
#
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

#
# Zero the image for the graph so it is display blank to start with.
# 
import shutil
shutil.copy("blank.png","classified.png")

# Dummy call to overcome problems with matplotlib changing the window size on first call
plt.figure(figsize=(10,10))

Builder.load_file("buttons.kv")

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

class ClassifyButton(Button):
    
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        self.model = load_model('MNIST_model.h5')
        
        print("")
        print(">>>>>>>>>")
        print("Model summary")
        print( self.model.summary() )
        print("")
        print("Model layers")
        for layer in self.model.layers:
            print(layer.name, layer.input.shape, layer.output.shape)
        print("<<<<<<<<<")          
        print("")
        #plot_model(self.model, to_file='model.png')
        #self.model = load_model('TFKeras.h5')

    def classify_image(self):
        Custom_image_dir = './'
        Custom_image = "userDrawn.png"
        img = image.load_img('%s/%s' %(Custom_image_dir, Custom_image),  target_size=(28, 28), color_mode="grayscale")
        imgRGB = image.load_img('%s/%s' %(Custom_image_dir, Custom_image),  target_size=(28, 28))
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
        
class MainApp(App):

    def build(self):
        self.title = 'Convolutional Neural Network, handwritten number classification'
        return Container()

if __name__ == "__main__":
    Window.size = (960,601)
    app = MainApp()
    app.run()