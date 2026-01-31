import cv2
import numpy as np
from PIL import Image

CONF_Threshold = 0.5

Config_file = "./datafile.pbtxt"
Frozen_Model = "./frozen_inference_graph.pb"
Class_file = "./objects.names"
Image_file = "./multiobjimg.png"

net = cv2.dnn.readNetFromTensorflow(Frozen_Model, Config_file)

with open(Class_file, "rt") as fpt:
    class_labels = fpt.read().rstrip('\n').split('\n')

image = Image.open(Image_file)
img = np.array(image)

if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(
    img, 1.0 / 127.5, (320, 320),
    (127.5, 127.5, 127.5),
    crop=False,
    swapRB=True
)

net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > CONF_Threshold:
        class_id = int(detections[0, 0, i, 1])

        label = class_labels[class_id - 1]

        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        x1, y1, x2, y2 = box.astype("int")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            img,
            f"{label} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

cv2.imshow("Detected Objects", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
