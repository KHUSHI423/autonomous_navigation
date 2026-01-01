import cv2
import numpy as np

# Cross-platform TFLite import
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


class ObjectDetector:
    """
    EfficientDet Lite 0 detector (raw output parsing).
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_height = None
        self.input_width = None

        # COCO labels (first 90)
        self.labels = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
            "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
            "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
            "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
            "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
            "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
            "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
            "remote","keyboard","cell phone","microwave","oven","toaster","sink",
            "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
            "toothbrush"
        ]

    def load_model(self):
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        _, self.input_height, self.input_width, _ = self.input_details[0]["shape"]

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32) / 255.0

    def detect(self, frame):
        if self.interpreter is None:
            self.load_model()

        input_tensor = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        # Raw outputs
        class_scores = self.interpreter.get_tensor(self.output_details[0]["index"])[0]  # [19206, 90]
        boxes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]         # [19206, 4]

        detections = []

        for i in range(class_scores.shape[0]):
            class_id = int(np.argmax(class_scores[i]))
            score = float(class_scores[i][class_id])

            if score >= self.confidence_threshold:
                ymin, xmin, ymax, xmax = boxes[i]
                label = self.labels[class_id] if class_id < len(self.labels) else "object"

                detections.append({
                    "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)],
                    "score": score,
                    "class_name": label
                })

        return detections
