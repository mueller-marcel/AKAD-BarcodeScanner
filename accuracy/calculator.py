def calculate_accuracy(scan_results : list[bool]) -> float:
    """
    Calculates the accuracy of the barcode scanner based on a list of boolean values
    representing the scan results. A positive scan result is true if the scanned barcode matches
    the code from the file, otherwise false. The accuracy is returned as percentage.
    :param scan_results: The list of boolean values representing the scan results
    """

    return sum(scan_results) / len(scan_results) * 100