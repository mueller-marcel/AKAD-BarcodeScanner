from pathlib import Path
from accuracy.Calculator import Calculator
from barcode.Scanner import Scanner

if __name__ == '__main__':

    # Create a list to store scan results
    scan_results : list[bool] = []

    # Create a dictionary to with the file pairs
    text_image_pairs : dict[str, str] = {}

    # Define base directory for dataset 1
    directory = Path("dataset")

    # Get the image files
    jpg_files = [file for file in directory.glob("*.jpg")]
    text_files = [file for file in directory.glob("*.txt")]

    # Search for corresponding text file and store the text and the image file in a dictionary
    for text_file in text_files:
        text_file_stem = text_file.stem
        for jpg_file in jpg_files:
            if text_file_stem in jpg_file.name:
                text_image_pairs[str(text_file)] = str(jpg_file)

    # Iterate over file pairs and scan the image
    for text_file, image_file in text_image_pairs.items():
        print(f"Scan {image_file} and compare it with {text_file}")
        result = Scanner.scan_barcode(image_file, text_file)
        if result:
            print(f"The barcode from {image_file} was correctly scanned.")
        else:
            print(f"The barcode from {image_file} was not correctly scanned.")

        scan_results.append(result)

    # Calculate the accuracy based on the scan results
    accuracy : float = 0.0
    try:
        accuracy = Calculator.calculate_accuracy(scan_results)
    except ZeroDivisionError:
        print("Die Liste ist leer. Eine Division durch 0 ist nicht erlaubt.")

    print(f"Die Genauigkeit des Scanners beträgt {accuracy} %")