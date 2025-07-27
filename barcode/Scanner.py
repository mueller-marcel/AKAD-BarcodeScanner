from barcode.Decoder import Decoder
from barcode.Detector import Detector
import cv2

class Scanner:

    def __init__(self):
        """
        Constructs the scanner class that is in charge of preprocessing, detecting and decoding the barcode
        """

    @staticmethod
    def scan_barcode(image_file: str, text_file: str, rotate_barcode: bool) -> bool:
        """
        Scans the image for a barcode and detects it.
        Compares the value of the scanned barcode with the original value from the text file
        :param image_file: The file path to the image
        :param text_file: The file path to the text file
        :param rotate_barcode: Rotates the barcode 180 degrees after cropping since the barcode rotation can be wrong
        """

        # Read the image
        image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

        if image is None:
            return False

        # Detect the barcode
        detector = Detector()
        cropped_barcode = detector.detect_barcode(image, rotate_barcode)

        if cropped_barcode is None:
            return False

        # Binarize the image
        _, binary_image = cv2.threshold(cropped_barcode, 127, 255, cv2.THRESH_BINARY)

        # Decode the barcode
        decoder = Decoder()
        digits = decoder.decode_barcode(binary_image)

        if digits is None:
            return False

        # Read the original barcode from the corresponding text file
        with open(str(text_file), "r") as file:
            original_digits = file.read()

            if original_digits == digits:
                return True

        return False
