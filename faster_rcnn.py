# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
%matplotlib inline
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageFont, ImageDraw
import json
import colorsys
import matplotlib.pylab as plt

print(tf.__version__)

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
with open('/content/image_info_test2017.json','r') as r:
    js = json.loads(r.read())
#js.keys()
labels = {i['id']:i['name'] for i in js['categories']}
print(labels)
print(len(labels))

hsv_tuples = [(x/len(labels), 0.8, 0.8) for x in range(len(labels))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
conf = 0.2

# Read and preprocess an image.
img = cv2.imread('/content/ttt.jpg')

# Read the graph.
with tf.io.gfile.GFile('/content/drive/MyDrive/Colab Notebooks/Computer Vision Repos/weights/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef() #tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.compat.v1.Session() as sess: #tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    orig = np.copy(img)

    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv2.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})


    #print(len(out))
    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    print(num_detections)
    #print(out[1].shape) # prob
    #print(out[2].shape) # bounding box
    #print(out[3].shape) # class_id

    for i in range(num_detections):
        idx = int(out[3][0][i])
        #print(class_id)
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > conf:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows

        # draw the prediction on the image
        label = "{}: {:.2f}%".format(labels[idx], score * 100)
        color = tuple([int(255*x) for x in colors[idx]])
        y = y - 15 if y - 15 > 15 else y + 15
        pil_im = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        thickness = (img.shape[0] + img.shape[1]) // 300
        font = ImageFont.truetype("/content/Arial.ttf", 15)
        draw = ImageDraw.Draw(pil_im)
        label_size = draw.textsize(label, font)
        if y - label_size[1] >= 0:
            text_origin = np.array([x, y - label_size[1]])
        else:
            text_origin = np.array([x, y + 1])
        for i in range(thickness):
            draw.rectangle([x + i, y + i, right - i, bottom - i], outline=color)
        draw.rectangle([tuple(text_origin), tuple(text_origin +  label_size)], fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

fig = plt.figure(figsize=(20,20))
plt.imshow(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Objects detected with Faster-RCNN', size=25)
plt.show()