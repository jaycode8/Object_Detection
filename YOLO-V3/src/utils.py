
import cv2

class YoloNet:
    def __init__(self, config_file, weights, labels):
        self.net = cv2.dnn.readNet(weights, config_file)
        self.labels = labels
        self.classes = []

    def build_model(self):
        with open(self.labels, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        return self.net, self.classes, output_layers


