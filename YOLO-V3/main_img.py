
import cv2
from src.utils import YoloNet
import numpy as np

def detect_image(net, classes, output_layers, path):
    img = cv2.imread(path)
    height, width, channels = img.shape
    #detect objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    conf_threshold = 0.5
    nms_threshold = 0.4
    #process the detection
    boxes = []
    class_ids = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                #scale bounding box coordinates
                box = detection[0:4]*np.array([width, height, width, height])
                cx, cy, w, h = box.astype('int')
                #calculate top-left coordinates
                x = int(cx - (w/2))
                y = int(cy - (h/2))
                # Add detection information to lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y, int(w), int(h)])
    # apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences,conf_threshold, nms_threshold)

    # draw bounding boxes and display results
    for i in indices:
        #i = i[0]
        x,y,w,h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = (255,0,0)
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #display results
    cv2.imshow('Object Detection', img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    config_file = './darknet/yolov3.cfg'
    weights = './darknet/yolov3.weights'
    label_file = './darknet/coco.names'
    img_path = '../resources/img1.jpeg'
    yolo = YoloNet(config_file, weights, label_file)
    net, classes, output_layers = yolo.build_model()
    detect = detect_image(net, classes, output_layers, img_path)
    #print(labels)
