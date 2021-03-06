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

### FHD displatys
LINEWIDTH = 15
### UHD(4K) displays
#LINEWIDTH = 30

#
# Kivy
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
# Tensorflow and Keras
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
from kivy.properties import StringProperty

#
# Utils
#
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os import listdir 
import matplotlib.cm as cm
from PIL import Image, ImageFont, ImageDraw
import shutil
import uuid

#Disable developer warning messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#
# Zero the image for the graph so it is display blank to start with.
# 
shutil.copy("blank.png","classified.png")
shutil.copy("blank.png","cnnInput.png")
shutil.copy("blank.png","saliency.png")
shutil.copy("glyphNoData_crp.png", "currentGlyph.png")

# Dummy call to overcome problems with matplotlib changing the window size on first call
plt.figure(figsize=(10,10))

Builder.load_file("buttons.kv")

#
# Prepare CNN'a
# By chance model 00 and model 01 the final layer have the same name: dense_2
#
num_of_models = 3
global_model=[None]*num_of_models
global_model_name=[None]*num_of_models
global_saliency=[None]*num_of_models
glb_indx = 0
print("")
print(">>>>>>>>>")
print("Network 00 loading")
print("")
global_model_name[0] = 'MNIST_model_00.h5'
global_model[0] = load_model(global_model_name[0])
global_saliency[0] = load_model(global_model_name[0])
layer_idx = utils.find_layer_idx(global_saliency[0], 'dense_2')
# Swap softmax with linear
global_saliency[0].layers[layer_idx].activation = keras.activations.linear
global_saliency[0] = utils.apply_modifications(global_saliency[0])

print("")
print(">>>>>>>>>")
print("Network 01 loading")
print("")
global_model_name[1] = 'MNIST_model_01.h5'
global_model[1] = load_model(global_model_name[1] )
global_saliency[1] = load_model(global_model_name[1] )
layer_idx = utils.find_layer_idx(global_saliency[1], 'dense_2')
# Swap softmax with linear
global_saliency[1].layers[layer_idx].activation = keras.activations.linear
global_saliency[1] = utils.apply_modifications(global_saliency[1])

print("")
print(">>>>>>>>>")
print("Network 02 loading")
print("")
global_model_name[2] = 'MNIST_model_04.h5'
global_model[2] = load_model(global_model_name[2])
global_saliency[2] = load_model(global_model_name[2])
layer_idx = utils.find_layer_idx(global_saliency[2], 'dense_2')
# Swap softmax with linear
global_saliency[2].layers[layer_idx].activation = keras.activations.linear
global_saliency[2] = utils.apply_modifications(global_saliency[2])

print("")
print("Networks loaded")
print("<<<<<<<<<")
print("")

# Globals for current result: set current prediction to -1
archive_dir = "./MCHCArchive"
curr_prd = -1
curr_conf = 0
curr_drawn = None
curr_img = None

class Container(BoxLayout):
    display = ObjectProperty()

class LabelDraw(Label):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 0)
            d = 30.
            touch.ud['line'] = Line(points=(touch.x, touch.y),width = LINEWIDTH)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def clear_canvas(self):
        self.canvas.clear()

    def save_canvas(self):
     self.export_as_image().save("userDrawn.png", flipped=False)   

class ClrButton(Button):
    pass

class ClrAllButton(Button):
    pass

class Model00Button(Button):
    pass

class Model01Button(Button):
    pass

class Model02Button(Button):
    pass

class HC_Button(Button):
    def print_me(self,humanLabel):
        global curr_pr
        global curr_conf
        global curr_img
        global curr_drawn
        global archive_dir
        global glb_indx
        img_uid = uuid.uuid4().hex
        print( "My human classification is : ", humanLabel)
        print( "My machine classification is :", curr_prd, curr_conf, global_model_name[glb_indx] )

        archive_header = "MC, HC, PR, MDL, ID"
        print( archive_header)
        archive_data = str(curr_prd)+","+str(humanLabel)+","+ str(curr_conf)+","+global_model_name[glb_indx] +","+str(img_uid)
        print(archive_data)
        archive_file = archive_dir+"/"+"NCLimg_"+uuid.uuid4().hex+"_MC_"+str(curr_prd)+"_HC_"+str(humanLabel)
        print( "Folder and name is: ", archive_file)

        f = open(archive_file+".txt", "a")
        print(archive_header, file=f)
        print(archive_data, file=f)
        f.close()

        curr_img.save( archive_file+"_img.png")
        curr_drawn.save( archive_file+"_drw.png")

class HI_00_Button(HC_Button):
    pass
class HI_01_Button(HC_Button):
    pass
class HI_02_Button(HC_Button):
    pass
class HI_03_Button(HC_Button):
    pass
class HI_04_Button(HC_Button):
    pass
class HI_05_Button(HC_Button):
    pass
class HI_06_Button(HC_Button):
    pass
class HI_07_Button(HC_Button):
    pass
class HI_08_Button(HC_Button):
    pass
class HI_09_Button(HC_Button):
    pass


class SaliencyButton(Button):
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        self.model = global_saliency[glb_indx]

        print("")
        print(">>>>>>>>>")
        print("Saliency network loaded")
        print("")
        
    def setModel( self, num ):
        global glb_indx

        glb_indx = num
        self.model = global_saliency[glb_indx]
        print( "Saliency network index is now: " + str(glb_indx) )


    def saliency_image(self):
        global glb_indx

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
        # Swap softmax with linear <<Done>>
        #self.model.layers[layer_idx].activation = keras.activations.linear
        #self.model = utils.apply_modifications(self.model)

        #Check we can still categorise the image.
        prediction =self.model.predict(img_data)
        prediction_category = np.argmax(prediction[0])
        print("Predicted numeral modified model: ", prediction_category)
        print( "The prediction for saliency: "+ str(prediction.argmax() ) + " by network: " +str(glb_indx))

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
        theTitle = 'Saliency, predicts: '+str(prediction.argmax())+ ' by network: ' + str(glb_indx)
        plt.title(theTitle,fontsize=32 )
        plt.imshow(grad_eval_by_hand,cmap="coolwarm",alpha=0.8)
        plt.savefig('saliency.png')
        plt.close("all")

class ClassifyButton(Button):
    
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        self.model = global_model[glb_indx]
        
    def setModel( self, num ):
        global glb_indx

        glb_indx = num
        self.model = global_model[glb_indx]
        print( "Classify network index is now: " + str(glb_indx) )

    def generateGlyph( self ):
        global curr_prd
        global curr_conf
        if (curr_prd == -1 ):
            shutil.copy("glyphNoData_crp.png","currentGlyph.png")
        else:
            stemName = ['G','F','E','D','C','B','A']
            # offset confidence so 0.3 is lowest then in steps of 0.1 upwards
            if curr_conf < 0.3:
                nameIndx = 0
            else:
                nameIndx = int(10.0*(curr_conf-0.3))
            nameIndx = np.clip(nameIndx,0, 6)
            filename = '.\\VisEntGlyphs\\' + stemName[nameIndx] + '_crp_over.png'
            source_img = Image.open(filename).convert("RGBA")
            draw = ImageDraw.Draw(source_img)
            font = ImageFont.truetype("arial.ttf", 240)
            num = str(curr_prd)
            xD,yD=draw.textsize(num, font=font)
            draw.text((320-(xD/2),240-(yD/2)), num, fill=(255,255,0,255), font=font )
            font = ImageFont.truetype("arial.ttf", 32)
            txt = "High ------------- probability ------------- Low" 
            draw.text((20,525), txt, fill=(51,51,51,255), font=font )
            source_img.save("currentGlyph.png")

    def clear_classify(self):
        global curr_prd
        global curr_conf
        shutil.copy("blank.png","classified.png")
        shutil.copy("glyphNoData_crp.png", "currentGlyph.png")
        shutil.copy("blank.png","cnnInput.png")
        shutil.copy("blank.png","saliency.png")
        curr_prd = -1
        curr_conf = 0

    def classify_image(self):
        global glb_indx
        global curr_prd
        global curr_conf
        global curr_img
        global curr_drawn

        Custom_image_dir = './'
        Custom_image = "userDrawn.png"
        img = image.load_img('%s/%s' %(Custom_image_dir, Custom_image),  target_size=(28, 28), color_mode="grayscale")

        # Save images for archiving
        curr_img = img
        curr_drawn = image.load_img('%s/%s' %(Custom_image_dir, Custom_image))

        plt.figure(figsize=(10,10))
        plt.title('Image as input to CNN',fontsize=32)
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
        print( "The prediction is: " + str(prediction.argmax()) + " by network: " +str(glb_indx))

        curr_prd = prediction.argmax()
        curr_conf = prediction[0][curr_prd]
        print( curr_prd, curr_conf,prediction[0])
        plt.figure(figsize=(10,10))
        index = np.arange(10)
        my_cmap = cm.get_cmap('winter')
        plt.bar(index, prediction[0],color=my_cmap(prediction[0]))
        theTitle = 'Predicted: ' + str(np.argmax(prediction[0])) + ' by network: ' +str(glb_indx)
        plt.title(theTitle,fontsize=32 )
        plt.xticks(index)
        plt.yticks(np.arange(0, 1.1, step = 0.1))
        #plt.show(block=False)
        plt.savefig('classified.png')
        plt.close("all")

class modelLabel( Label ):

    def set_txt_lbl(self):
        global global_model_name
        global glb_indx
        txt = global_model_name[glb_indx]
        print("Label: ", txt)
        self.text = txt

        
class MainApp(App):

    def build(self):
        self.title = 'Convolutional Neural Network, handwritten number classification'
        return Container()

if __name__ == "__main__":
    Window.size = (1600,960)
    app = MainApp()
    app.run()