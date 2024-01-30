import numpy as np
import cv2
import matplotlib.pylab as plt

def non_max_suppression(boxes, scores, threshold):
    """
    NMS (non-max suppressuion) to ignore redundant & overlapping bounding boxes
    return: boxes_keep_index
    """
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)

def compute_iou(box, boxes, box_area, boxes_area):
    """
    compute intersection over union metric (iou) for a given box against other boxes
    return: ious
    """
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the  pulled target-x minus the pushed origin-x
    # multiplying pulled target-y minus the pushed origin-y 
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious


def main():
    #read the image & initilize the HOG
    img = cv2.imread('/content/walk.png')
    hog = cv2.HOGDescriptor()

    #set the SVM detector (default people detector)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    #NMS (non-max suppressuion) to ignore redundant & overlapping bounding boxes

    #compute intersection over union metric (iou) for a given box against other boxes
    
    #detect pedestrains & obtain the bounding boxes
    found_bounding_boxes, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.01, useMeanshiftGrouping=True)
    # print(len(found_bounding_boxes))

    # #apply non-max supression
    boxIndices = non_max_suppression(found_bounding_boxes, weights.ravel(), threshold=0.2)
    found_bounding_boxes = found_bounding_boxes[boxIndices,:]

    print(len(found_bounding_boxes)) # number of boundingboxes

    #draw the bounding boxes after NMS
    img_with_raw_bboxes = img.copy()
    for (hx, hy, hw, hh) in found_bounding_boxes:
        cv2.rectangle(img_with_raw_bboxes, (hx, hy), (hx + hw, hy + hh), (0, 0, 255), 2)
    plt.figure(figsize=(20, 12))
    img_with_raw_bboxes = cv2.cvtColor(img_with_raw_bboxes, cv2.COLOR_BGR2RGB)
    plt.imshow(img_with_raw_bboxes, aspect='auto'), plt.axis('off')
    plt.title('Bounding boxes found by HOG after NMS with meanshift group', size=20)
    plt.show()


if __name__ == "__main__":
  main()