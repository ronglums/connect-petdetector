import argparse
import sys
import time
import numpy as np
import tensorflow as tf
import json
import os
from base64 import b64encode, b64decode
from azureml.core.model import Model
from azureml.core import Workspace

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def init():
    global graph
    model_path = Model.get_model_path(model_name='pet-detector')
    graph = load_graph(os.path.join(model_path, 'output_graph.pb'))
    
def run(raw_data):
    input_name = 'import/input'
    output_name = 'import/final_result'
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    # Turn raw data (json) into a JPEG image
    base64_string = json.loads(raw_data)['image']
    base64_bytes = b64decode(base64_string)
    image_reader = tf.image.decode_jpeg(base64_bytes, channels=3, name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander,[224,224])




    


    # Run a Tensorflow session to make predications on the image
    sess = tf.Session()
    sess.run(resized)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: image})
        end=time.time()
        
    results = np.squeeze(results)

    # Return the top k species and their probability
    top_k = results.argsort()[-5:][::-1]
    model_path = Model.get_model_path(model_name='pet-detector')
    labels = load_labels(os.path.join(model_path, 'output_labels.txt'))
    predictions = []
    for i in top_k:
        predictions.append('%s (score=%0.5f)' % (labels[i], results[i]))
    
    struct = {
        'evaluation_time': 'Evaluation time (1-image): %.3f' % end - start,
        'predictions': predictions
    }
    
    return json.dumps(struct)