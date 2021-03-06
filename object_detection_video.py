### https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
import numpy as np
import cv2
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

tf.reset_default_graph()
tf.keras.backend.clear_session()
## for GPU
config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

# Path to CheckPoint (frozen inference graph)
#MODEL_NAME ='faster_rcnn_inception_v2_coco_2018_01_28'
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
FROZEN_GRAPH_FILE = MODEL_NAME+'/frozen_inference_graph.pb'
LABEL_MAP_FILE = 'data/mscoco_label_map.pbtxt'

VIDEO_FILE = 'test.mov'
NUM_CLASSES = 90

# Load label map (indices to category names)
label_map = label_map_util.load_labelmap(LABEL_MAP_FILE)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load Model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(FROZEN_GRAPH_FILE, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (for image classifier)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# score is level of confidence
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(VIDEO_FILE)
ret = video.set(3,1280)
ret = video.set(4,720)

while(True):
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    ret, frame = video.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform Detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw Results of Detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    # Show Frame
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()