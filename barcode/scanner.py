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

        # Calculate the gradients
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)

        # Blur the image to reduce noise
        blurred = cv2.blur(gradient, (9, 9))

        # Threshold the image
        _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        # Morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Erode and dilate
        closed = cv2.erode(closed, kernel, iterations=4)
        closed = cv2.dilate(closed, kernel, iterations=4)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours were found, exit
        if not contours:
            print("No barcode found")
            exit()

        # Sort the contours by area, keeping the largest one
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Compute the bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = box.astype(int)

        # Draw a bounding box around the detected barcode
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

        cv2.imshow("Barcode", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return gray

    @staticmethod
    def __detect_barcode(binary_image: np.ndarray) -> np.ndarray | None:
        """
        Detect the barcode from the image and returns the barcode from the image.
        :param binary_image: The preprocessed image. The image is expected to be binary.
        :return: The region of interest (ROI), if a barcode is found, else None
        """

        return binary_image