import cv2
import numpy as np
import matplotlib.pylab as plt
from PIL import Image, ImageDraw, ImageFont
import colorsys
from random import shuffle
from google.colab.patches import cv2_imshow


#initilize all of the parameters
confThreshold = 0.5 #confidence threshold
nmsThreshold = 0.4 #nms threshold
inpWidth = 416 #net width (input image)
inpHeight = 416 #net height (input image)

#load the names of the object classes
classes_file = "/content/coco_classes.txt";
classes = None
with open(classes_file, 'rt') as f:
  classes = f.read().rstrip('\n').split('\n')

#create unique colors related to each object class
hsv_tuples = [(x/len(classes), x/len(classes), 0.8) for x in range(len(classes))]
shuffle(hsv_tuples)
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

def get_output_layers(net):
    """
    get the output layer names
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, left, top, right, bottom):
    """
    draw the bounding box around a predicted object with label & confidence value
    """
    # Draw a bounding box.
    label = "{}: {:.2f}%".format(classes[class_id], confidence * 100)
    color = tuple([int(255*x) for x in colors[class_id]])
    top = top - 15 if top - 15 > 15 else top + 15
    pil_im = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
    thickness = (img.shape[0] + img.shape[1]) // 300
    font = ImageFont.truetype("/content/verdana.ttf", 25) 
    draw = ImageDraw.Draw(pil_im)  
    label_size = draw.textsize(label, font)
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)
    draw.rectangle([tuple(text_origin), tuple(text_origin +  label_size)], fill=color)
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw
    img = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  
    
    return img    

# Remove the bounding boxes with low confidence using non-maxima suppression
def post_process(img, outs):
    heighteight = img.shape[0]
    widthidth = img.shape[1]

    class_ids = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                center_x = int(detection[0] * widthidth)
                center_y = int(detection[1] * heighteight)
                width = int(detection[2] * widthidth)
                height = int(detection[3] * heighteight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
      for j in range(len(indices)):
        i = indices[j]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        img = draw_bounding_box(img, class_ids[i], confidences[i], left, top, left + width, top + height)
        
    return img    

def main():

    # Give the configuration and weight files for the model and load the network using them.
    model_configuration = "/content/yolov3.cfg"
    
    # to download weights --> !wget https://pjreddie.com/media/files/yolov3.weights
    model_weights = "/content/yolov3.weights"

    # load the pretrained deep NN with darknet confg & weight files
    net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    img_path = "/content/table.png"

    img = cv2.imread(img_path)

    orig = np.copy(img)
    # Create a 4D blob from a img.
    blob = cv2.dnn.blobFromImage(img, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_output_layers(net))

    # Remove the bounding boxes with low confidence
    img = post_process(img, outs)

    fig = plt.figure(figsize=(20,15))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('YoloV3 Objects detection using opencv', size=20)
    plt.show()
if __name__ == "__main__":
    main()