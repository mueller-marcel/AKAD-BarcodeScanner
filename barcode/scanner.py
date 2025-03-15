import cv2
import numpy as np

class BarcodeScanner:
    def __init__(self):
        """
        Initialize the barcode scanner, training the model to detect barcodes.
        """

        print("Hello")

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

        # Preprocess the image
        preprocessed_image = self.__preprocess_image(image)

        # Detect the barcode, if none was found the algorithmus terminates with false
        barcode_image = self.__detect_barcode(preprocessed_image)
        if barcode_image is None:
            return False

        return True

    @staticmethod
    def __preprocess_image(image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the image and returns the preprocessed image. Extracts the barcode from the image.
        :param image: The image to preprocess
        :return: The preprocessed image of the barcode only
        """

        # Convert the image to gray image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a binary image
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        return binary_image

    @staticmethod
    def __detect_barcode(binary_image: np.ndarray) -> np.ndarray | None:
        """
        Detect the barcode from the image and returns the barcode from the image.
        :param binary_image: The preprocessed image. The image is expected to be binary.
        :return: The region of interest (ROI), if a barcode is found, else None
        """

        return binary_image