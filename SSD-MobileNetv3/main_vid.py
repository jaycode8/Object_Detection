
from src.utils import MobileNet
import cv2

def video_detection(model, class_labels):
    cap = cv2.VideoCapture('.././resources/vid.mp4')
    while True:
        _, frame = cap.read()
        ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.55)

        if (len(ClassIndex) != 0):
            for ClassInd, conf, boxes in zip(ClassIndex, confidece, bbox):
                if (ClassInd <= 80):
                    cv2.rectangle(frame, boxes, (255,0,0), 1)
                    cv2.putText(frame, class_labels[ClassInd-1], (boxes[0]+10, boxes[1]+40), cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0,255,0), thickness=3)
        cv2.imshow('Video', frame)
        key = cv2.waitKey(100)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config_file = './data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = './data/frozen_inference_graph.pb'
    labels = './data/labels.txt'
    mn = MobileNet(config_file, frozen_model,labels)
    model, lbs = mn.build_model()
    video_detection(model, lbs)





