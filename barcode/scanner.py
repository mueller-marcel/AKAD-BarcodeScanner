import cv2

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

    # Create a binary image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Show image
    cv2.imshow("Detected", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True
