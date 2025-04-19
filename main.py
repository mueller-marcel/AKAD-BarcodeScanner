from pathlib import Path
from accuracy.calculator import calculate_accuracy
from barcode.Scanner import Scanner

if __name__ == '__main__':

    # Instantiate the barcode scanner
    scanner = Scanner()

    # Create a list to store scan results
    scan_results : list[bool] = []

    # Create a dictionary to with the file pairs
    text_image_pairs : dict[str, str] = {}

    # Define base directory for dataset 1
    directory = Path("Dataset1")

    # Get the image files
    jpg_files = [file for file in directory.glob("*.jpg")]
    text_files = [file for file in directory.glob("*.txt")]

    # Search for corresponding text file and store the text and the image file in a dictionary
    for text_file in text_files:
        text_file_stem = text_file.stem
        for jpg_file in jpg_files:
            if text_file_stem in jpg_file.name:
                text_image_pairs[text_file] = jpg_file

    # Iterate over file pairs and scan the image
    for text_file, image_file in text_image_pairs.items():
        print(f"Scanne {image_file} und vergleiche sie mit {text_file}")

        result = scanner.scan_barcode(image_file, text_file)
        scan_results.append(result)

    # Calculate the accuracy based on the scan results
    accuracy : float = 0.0
    try:
        accuracy = calculate_accuracy(scan_results)
    except ZeroDivisionError:
        print("Die Liste ist leer. Eine Division durch 0 ist nicht erlaubt.")

    print(f"Die Genauigkeit des Scanners beträgt {accuracy} %")