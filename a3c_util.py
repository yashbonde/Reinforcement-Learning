# import the dependencies
import numpy as np # matrix math
import tensorflow as tf # Ml
import cv2 # image processing
import moviepy.editor as mpy # making GIF
import scipy.signal # discount factor

def make_gif(images, fname, duration = 2, true_image = False, salience = False, salIMGS = None):
    # this function makes GIF
    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]
            
        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)
        
    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS)/duration*t)]
        except:
            x = salIMGS[-1]
        return x
    
    clip = mpy.VideoClip(make_frame, duration = duration)
    if salience:
        mask = mpy.VideoClip(make_mask, ismask = True, duration = duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps = len(images)/duration, verbose = False)
    else:
        clip.write_gif(fname, fps = len(images)/duration, verbose = False)

def update_target_graph(from_scope, to_scope):
    # function to copy one set of worker variables to the another 
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    # discounting function used to calculate discounted returns
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_frame(image, output_shape = (80, 80)):
    # crop the image to proper bounding box
    image_croped = image[34: 160, :]
    image_resized = cv2.resize(image, output_shape)
    image_float = image_resized.astype(float)
    return (image_float-127.5)/255.0
    