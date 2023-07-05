
# -------------------------- this class sets up the SSD-MobileNet3 configuration

import cv2

class MobileNet:
    def __init__(self,config,frozen_model,label):
        self.model = cv2.dnn_DetectionModel(frozen_model, config) 
        self.labels = label
        self.class_labels = []

    def build_model(self):
        with open(self.labels, 'rt') as f:
            self.class_labels = [line.strip() for line in f.readlines()]
        self.model.setInputSize(320,320)
        self.model.setInputScale(1.0/127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)
        return self.model, self.class_labels

# if __name__ == '__main__':
#     config_file = './data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#     frozen_model = './data/frozen_inference_graph.pb'
#     labels = './data/labels.txt'
#     mn = MobileNet(config_file, frozen_model,labels)
#     print(mn.build_model())
