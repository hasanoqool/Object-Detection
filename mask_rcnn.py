import cv2
import numpy as np
import os.path
import sys
import random
import matplotlib.pylab as plt

print(cv2.__version__)

# Initialize the parameters
conf_threshold = 0.5  # Confidence threshold
mask_threshold = 0.3  # Mask threshold

# Draw the predicted bounding box, colorize and show the mask on the image
def draw_box(img, class_id, conf, left, top, right, bottom, class_mask):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), (255, 178, 50), 3)
    
    # Print a label of class.
    label = '%.2f' % conf
    if classes:
        assert(class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)
    
    # Display the label at the top of the bounding box
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv2.rectangle(img, (left, top - round(1.5*label_size[1])), (left + round(1.5*label_size[0]), top + baseline), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    # Resize the mask, threshold, color and apply it on the image
    class_mask = cv2.resize(class_mask, (right - left + 1, bottom - top + 1))
    mask = (class_mask > mask_threshold)
    roi = img[top:bottom+1, left:right+1][mask]

    # color = colors[class_id%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    color_index = random.randint(0, len(colors)-1)
    color = colors[color_index]

    img[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    #im2, 
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img[top:bottom+1, left:right+1], contours, -1, color, 3, cv2.LINE_8, hierarchy, 100)

# For each img, extract the bounding box and mask for each detected object
def post_process(boxes, masks):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    num_classes = masks.shape[1]
    num_detections = boxes.shape[2]

    height = img.shape[0]
    width = img.shape[1]

    for i in range(num_detections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > conf_threshold:
            class_id = int(box[1])
            
            # Extract the bounding box
            left = int(width * box[3])
            top = int(height * box[4])
            right = int(width * box[5])
            bottom = int(height * box[6])
            
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(0, min(right, width - 1))
            bottom = max(0, min(bottom, height - 1))
            
            # Extract the mask for the object
            class_mask = mask[class_id]

            # Draw bounding box, colorize and show the mask on the image
            draw_box(img, class_id, score, left, top, right, bottom, class_mask)


# Load names of classes
classesFile = "/content/mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
   classes = f.read().rstrip('\n').split('\n')

# Give the textGraph and weight files for the model
textGraph = "/content/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
model_weights = "/content/drive/MyDrive/Colab Notebooks/Computer Vision Repos/weights/frozen_inference_graph_mask.pb";

# Load the network
net = cv2.dnn.readNetFromTensorflow(model_weights, textGraph);
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load the classes
colors_file = "/content/colors.txt";
with open(colors_file, 'rt') as f:
    colors_str = f.read().rstrip('\n').split('\n')
colors = [] #[0,0,0]
for i in range(len(colors_str)):
    rgb = colors_str[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

img = cv2.imread('/content/road.png')
    
print(img.shape)

orig = np.copy(img)
#cv2.imwrite('Mask-RCNN/input/img_' + str(i).zfill(4) + '.jpg', orig)

# Create a 4D blob from a img.
blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)

# Set the input to the network
net.setInput(blob)

# Run the forward pass to get output from the output layers
boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

# Extract the bounding box and mask for each of the detected objects
post_process(boxes, masks)

# Put efficiency information.
t, _ = net.getPerfProfile()
#label = 'Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a img : %0.0f ms' % abs(t * 1000.0 / cv2.getTickFrequency())
#cv2.putText(img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) 
plt.subplot(211), plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Original Image', size=20)
plt.subplot(212), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.axis('off'), plt.title('Objects detected with Mask-RCNN', size=20)
plt.show()