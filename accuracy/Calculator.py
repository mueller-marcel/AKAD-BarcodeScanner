class Calculator:

    @staticmethod
    def calculate_accuracy(scan_results : list[bool]) -> float:
        """
        Calculates the accuracy of the barcode scanner based on a list of boolean values
        representing the scan results. A positive scan result is true if the scanned barcode matches
        the code from the file, otherwise false. The accuracy is returned as percentage.
        :param scan_results: The list of boolean values representing the scan results
        """

        # Filter for none elements
        filtered_results = [res for res in scan_results if res is not None]

        # Divide by zero is not supported
        if not filtered_results:
            raise ZeroDivisionError("Die Liste ist leer. Eine Division durch 0 ist nicht erlaubt.")

        return sum(scan_results) / len(scan_results) * 100