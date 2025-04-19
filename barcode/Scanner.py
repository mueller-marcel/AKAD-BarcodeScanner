import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class Scanner:

    def scan_barcode(self, image_path: str, text_file_path: str) -> bool:
        """
        Retrieves the barcode from the image_path and compares it with the barcode text from
        the text_file_path. Returns a boolean indicating if the retrieved barcode matches the
        barcode from the text_file_path.
        :rtype: bool
        :param image_path: The path to the image in jpg format
        :param text_file_path: The path to the text file in txt format
        """

        # Load the image
        image = cv2.imread(image_path)

        # Detect the barcode, if none was found the algorithmus terminates with false
        barcode_image = self.__detect_barcode(image)
        if barcode_image is None:
            return False

        return True


    @staticmethod
    def __detect_barcode(image: np.ndarray) -> np.ndarray | None:
        """
        Detect the barcode from the image and returns the barcode from the image.
        :param image: The image as numpy array.
        :return: The region of interest (ROI), if a barcode is found, else None
        """

        # Get path to the model and initialize it
        path_to_model = os.path.join(Path.cwd(), "runs", "detect","train6", "weights", "best.pt")
        model = YOLO(path_to_model)

        # Detect the barcode
        results = model(image)

        # Draw rectangle around the barcode if found
        if len(results) > 0:
            for result in results:
                for box in result.boxes:

                    # Get the coordinates and draw rectangle
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show images with rectangles
        cv2.imshow("Barcodes", image)
        cv2.waitKey(0)

        return image
