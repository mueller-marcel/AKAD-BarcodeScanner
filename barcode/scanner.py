import cv2
import numpy as np

def scan_barcode(image_path: str, text_file_path: str) -> bool:
    """
    Retrieves the barcode from the image_path and compares it with the barcode text from
    the text_file_path. Returns a boolean indicating if the retrieved barcode matches the
    barcode from the text_file_path.
    :rtype: bool
    :param image_path: The path to the image in jpg format
    :param text_file_path: The path to the text file in txt format
    """

    # Load the image using grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    image = preprocess_image(image)

    return True

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image and returns the preprocessed image. Extracts the barcode from the image.
    :param image: The image to preprocess
    :return: The preprocessed image of the barcode only
    """

    # Create a binary image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Show image
    cv2.imshow("Detected", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image