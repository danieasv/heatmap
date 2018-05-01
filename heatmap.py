
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import transform
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Activation
from vis.utils import utils
import cv2
from pathlib import Path

tf.contrib.keras.preprocessing

FRAME_H = 64
FRAME_W = 64

pred_image = np.zeros((1, 64, 64, 3))

model = load_model('model_fc_dropout_460_riktig')

pathlist = Path('Film_VGG16_NVIDIA_460_1').glob('**/*.jpg')
for path in pathlist:
    # because path is object not strin
    path_in_str = str(path)
    name_of_img = path_in_str.strip('Film_VGG16_NVIDIA_460_1/')
    
    image = utils.load_img(path_in_str)
    shape = image.shape
    image_pred = image/255.-.5

    image = np.array(image)
    image_pred = np.array(image_pred)

    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)

    image_pred = image_pred[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image_pred = cv2.resize(image_pred,(64,64), interpolation=cv2.INTER_AREA)
   
    '''
    plt.figure()
    plt.subplot()
    plt.imshow(image)
    plt.show()
    '''

    #pred_image[0] = image_pred

    #pred = model.predict(pred_image)[0][0]
    #print('Predicted {}'.format(pred))
    '''
    from vis.visualization import visualize_saliency, overlay
    #titles = ['right steering', 'left steering', 'CenterCam']
    
    for i, modifier in enumerate(modifiers):
        heatmap = visualize_saliency(model, layer_idx=-1, filter_indices=None, seed_input=image_pred, grad_modifier=modifier) #, filter_indices=0
        plt.figure()

        # Overlay is used to alpha blend heatmap onto img.
        plt.imshow(overlay(image, heatmap, alpha=0.7))
        plt.show()
    '''
    from vis.visualization import visualize_cam
    #modifiers = [None, 'negate', 'small_values']
    modifiers = [None]
    for i, modifier in enumerate(modifiers):
        heatmap = visualize_cam(model, layer_idx=-1, filter_indices=None, seed_input=image_pred, grad_modifier=modifier) #, filter_indices=0
        plt.figure()
        # Overlay is used to alpha blend heatmap onto img.
        plt.imshow(overlay(image, heatmap, alpha=0.7))
        plt.show()
        plt.savefig('Heatmap'+'/'+name_of_img, bbox_inches='tight')
        break
        #cv2.imwrite(path_in_str, name_of_img)
        #print(path_in_str)

    break



