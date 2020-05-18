import os
import sys
import scipy
import numpy as np
import cv2
import tensorflow as tf
from helper import logits2image

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

graph = load_graph(sys.argv[1])
image_dir = sys.argv[2]

# DeepLabv3+ input and output tensors
image_input = graph.get_tensor_by_name('prefix/ImageTensor:0')
softmax = graph.get_tensor_by_name('prefix/SemanticPredictions:0')

# Create output directories in the image folder
if not os.path.exists(image_dir+'segmented_images_colored/'):
    os.mkdir(image_dir+'segmented_images_colored/')

image_dir_segmented_colored = image_dir+'segmented_images_colored/'

batch_sz = 16

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(graph=graph, config=config) as sess:
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(".png"):
            img = cv2.imread(os.path.join(image_dir, fname))
            img = cv2.resize(img, (1002, 303))
            img = np.expand_dims(img, axis=0)
            probs = sess.run(softmax, {image_input: img})
            img = tf.squeeze(probs).eval()
            img_colored = logits2image(img)
            img_colored = cv2.resize(img_colored, (1241, 376))
            cv2.imwrite(os.path.join(image_dir_segmented_colored+fname),cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))
            print(fname)