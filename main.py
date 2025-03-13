import sys
from accuracy.calculator import calculate_accuracy
from barcode.scanner import scan_barcode
from pathlib import Path

if __name__ == '__main__':

    # Create a list to store scan results
    scan_results : list[bool] = []

    # Create a dictionary to with the file pairs
    text_image_pairs : dict[str, str] = {}

    # Ask user for the dataset to be analyzed
    print("Welchen Datensatz möchten Sie analysieren? Geben Sie '1' oder '2' ein")
    user_selection = input()
    if user_selection == '1':
        print("Der erste Datensatz wird gescannt.")
        directory = Path("Dataset1")
    elif user_selection == '2':
        print("Der zweite Datensatz wird gescannt.")
        directory = Path("Dataset2")
    else:
        print("Die Eingabe konnte nicht verarbeitet werden. Das Programm wird beendet.")
        sys.exit()

    # Get the image files
    jpg_files = [file for file in directory.glob("*.jpg")]
    text_files = [file for file in directory.glob("*.txt")]

    # Search for corresponding text file and store the text and the image file in a dictionary
    for text_file in text_files:
        text_file_stem = text_file.stem
        for jpg_file in jpg_files:
            if text_file_stem in jpg_file.name:
                text_image_pairs[text_file.name] = jpg_file.name

    # Iterate over file pairs and scan the image
    for text_file, image_file in text_image_pairs.items():
        print(f"Scanne {image_file} und vergleiche sie mit {text_file}")

        result = scan_barcode(image_file, text_file)
        scan_results.append(result)

    # Calculate the accuracy based on the scan results
    accuracy : float = 0.0
    try:
        accuracy = calculate_accuracy(scan_results)
    except ZeroDivisionError:
        print("The scan results are empty. Divide by zero is not allowed.")

    print(f"Die Genauigkeit des Scanners beträgt {accuracy} %")