 #!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[11]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import match_object as match

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')


# ## Env setup

# In[12]:


# This is needed to display the images.
##get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object detection imports
# Here are the imports from the object detection module.

# In[13]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[14]:


# What model to download.
MODEL_NAME_1 = 'graph_VJTEYE_1'
MODEL_NAME_2 = 'ssd_mobilenet_v1_coco_2017_11_17'



# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH_1 = MODEL_NAME_1 + '/frozen_inference_graph.pb'
##PATH_TO_FROZEN_GRAPH_2 = MODEL_NAME_2 + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS_1 = os.path.join('pos_neg_2/train_data', 'labels_VJTEYE_1.pbtxt')
##PATH_TO_LABELS_2 = os.path.join('data', 'mscoco_complete_label_map.pbtxt')



# ## Download Model

# In[ ]:





# ## Load a (frozen) Tensorflow model into memory.

# In[15]:


detection_graph_1 = tf.Graph()
with detection_graph_1.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH_1, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
##detection_graph_2 = tf.Graph()
##with detection_graph_2.as_default():
##  od_graph_def = tf.GraphDef()
##  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH_2, 'rb') as fid:
##    serialized_graph = fid.read()
##    od_graph_def.ParseFromString(serialized_graph)
##    tf.import_graph_def(od_graph_def, name='')



# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[16]:


category_index_1 = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_1, use_display_name=True)
##category_index_2 = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_2, use_display_name=True)



# ## Helper code

# In[17]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[18]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 20)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


# In[19]:


def run_inference_for_single_image(image, graph):
  with graph.as_default():  
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
          
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
##      print(output_dict['detection_scores'].shape, " ", np.max(output_dict['detection_scores']))
##      print(output_dict['detection_classes'], " ", output_dict['detection_classes'].dtype)

  return output_dict
  


# In[20]:


for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  print('image_np ', image_np.shape)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  print("np_expanded ", image_np_expanded.shape)
  # Actual detection.
  output_dict_1 = run_inference_for_single_image(image_np_expanded, detection_graph_1)
##  output_dict_2 = run_inference_for_single_image(image_np_expanded, detection_graph_2)

  # Visualization of the results of a detection.
##  vis_util.visualize_boxes_and_labels_on_image_array(
##      image_np,
##      output_dict_1['detection_boxes'],
##      output_dict_1['detection_classes'],
##      output_dict_1['detection_scores'],
##      category_index_1,
####      instance_masks=output_dict_1.get('detection_masks'),
##      use_normalized_coordinates=True,
##      line_thickness=8)
  image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

  image_np = match.detect_signboard(output_dict_1, 0.5, image_np, 'pos3')

##  vis_util.visualize_boxes_and_labels_on_image_array(
##      image_np,
##      output_dict_2['detection_boxes'],
##      output_dict_2['detection_classes'],
##      output_dict_2['detection_scores'],
##      category_index_2,
##      instance_masks=output_dict_1.get('detection_masks'),
##      use_normalized_coordinates=True,
##      line_thickness=8)
##  plt.figure(figsize=IMAGE_SIZE)
##  plt.imshow(image_np)
##  plt.show()
  cv2.imshow("Detected image", image_np)
  cv2.waitKey(0)
cv2.destroyAllWindows()
# In[ ]:




