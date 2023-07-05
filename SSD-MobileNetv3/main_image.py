
from src.utils import MobileNet
import cv2

def picture_detections(model, class_labels):
    img = cv2.imread('.././resources/img1.jpeg')
    ClassIndex,confidece,bbox = model.detect(img, confThreshold=0.5)
    for ClassInd, conf, boxes in zip(ClassIndex, confidece, bbox):
        if (ClassInd <= 80):
            cv2.rectangle(img, boxes, (255,0,0), 1)
            cv2.putText(img, class_labels[ClassInd-1], (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,255,0), thickness=3)
    cv2.imshow('test', img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    config_file = './data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = './data/frozen_inference_graph.pb'
    labels = './data/labels.txt'
    mn = MobileNet(config_file, frozen_model,labels)
    model, lbs = mn.build_model()
    picture_detections(model, lbs)

