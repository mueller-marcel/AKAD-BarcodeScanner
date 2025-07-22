import cv2.barcode
import numpy as np

def detect_barcode(image: np.array) -> np.ndarray | None:
    """
    Detect the barcode from the image and returns the barcode from the image.
    :param image: The image as numpy array.
    :return: The region of interest (ROI), if a barcode is found, else None
    """

    # Detect the barcode
    detector = cv2.barcode.BarcodeDetector()
    ok, corners = detector.detect(image)

    if not ok or corners is None or len(corners) == 0:
        print("Kein Barcode erkannt.")
        return None

    # Corner points as integer NumPy points
    points = corners[0].astype(np.int32)

    # Create a rotated rectangle
    rect = cv2.minAreaRect(points)
    center, size, angle = rect[0], rect[1], rect[2]
    center = tuple(map(int, center))
    size = tuple(map(int, size))

    # Rotate image
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # Extract the rectangle
    cropped = cv2.getRectSubPix(rotated, size, center)

    # Rotate the image if needed
    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped
