from ultralytics import YOLO

class Trainer:

    @staticmethod
    def train_model(yolo_model_path : str) -> YOLO:
        """
        Train the yolo using the given training data.
        :param yolo_model_path: The path to the data yaml file pointing to training and test data
        """

        # Choose the small model to be trained using the data
        model = YOLO("yolov8s.pt")

        # Train the model
        model.train(data=yolo_model_path, epochs=50, batch=16, imgsz=640)

        # Validate the model
        model.val()

        # Export the model
        model.export(format="onnx")

        return model