import cv2
import numpy as np
import os,copy

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.6
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
GREEN  = (0,255,0)
RED    = (0,0,255)

def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    
    # Sets the input to the network.
    net.setInput(blob)
    
    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process(input_color_image, input_depth_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    centers = []
    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_color_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

                center_x = int(cx*x_factor)
                center_y = int(cy*y_factor)
                centers.append((center_x, center_y))
                    
    
    # Perform non maximum suppression to eliminate redundant, overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    select_box_ind = None
    select_center = None
    dis_min = 1000

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]   
        center = centers[i]
        # Draw bounding box.             
        cv2.rectangle(input_color_image, (left, top), (left + width, top + height), RED, 2*THICKNESS)
        # Draw center 
        # cv2.circle(input_color_image, center, 5, YELLOW, 2*THICKNESS)  

        # find the min distance 
        dis = input_depth_image[center[1]][center[0]]
        if dis<dis_min:
            dis_min = dis
            select_box_ind = i


    # Find the picking object 
    if select_box_ind is not None:
        select_center = centers[select_box_ind]
        box = boxes[select_box_ind]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]   
        cv2.rectangle(input_color_image, (left, top), (left + width, top + height), GREEN, 2*THICKNESS) 

    return input_color_image, select_center

def detect_result(input_color_image, input_depth_image, net):
    detections = pre_process(copy.deepcopy(input_color_image),net)
    detect_out = post_process(copy.deepcopy(input_color_image), input_depth_image, detections)
    return detect_out

modelWeights = os.path.dirname(__file__) + "/best_bin.onnx"
model = cv2.dnn.readNet(modelWeights)

# if __name__ == '__main__':
#     # Load image.
#     frame = cv2.imread('bin_9.jpg')
#     # Give the weight files to the model and load the network using       them.
#     modelWeights = "./weights/bin_picking.onnx"
#     net = cv2.dnn.readNet(modelWeights)
#     # Process image.
#     detections = pre_process(frame, net)
#     img = post_process(frame.copy(), detections)
#     """
#     Put efficiency information. The function getPerfProfile returns       the overall time for inference(t) 
#     and the timings for each of the layers(in layersTimes).
#     """
#     cv2.imshow('Output', img)
#     cv2.waitKey(0)