import numpy as np
from scipy.signal import find_peaks

class Decoder:

    def __init__(self):
        """
        Constructs an instance of the Decoder
        """

    _LEFT_EDGES_TO_DIGIT_MAPPING = {
        (1, 1, 2, 3): 0,
        (1, 2, 2, 2): 1,
        (2, 2, 1, 2): 2,
        (1, 1, 4, 1): 3,
        (2, 3, 1, 1): 4,
        (1, 3, 2, 1): 5,
        (4, 1, 1, 1): 6,
        (2, 1, 3, 1): 7,
        (3, 1, 2, 1): 8,
        (2, 1, 1, 3): 9,

        (3, 2, 1, 1): 0,
        (2, 2, 2, 1): 1,
        (2, 1, 2, 2): 2,
        (1, 4, 1, 1): 3,
        (1, 1, 3, 2): 4,
        (1, 2, 3, 1): 5,
        (1, 1, 1, 4): 6,
        (1, 3, 1, 2): 7,
        (1, 2, 1, 3): 8,
        (3, 1, 1, 2): 9,
    }

    _RIGHT_EDGES_TO_DIGIT_MAPPING = {
        (3, 2, 1, 1): 0,
        (2, 2, 2, 1): 1,
        (2, 1, 2, 2): 2,
        (1, 4, 1, 1): 3,
        (1, 1, 3, 2): 4,
        (1, 2, 3, 1): 5,
        (1, 1, 1, 4): 6,
        (1, 3, 1, 2): 7,
        (1, 2, 1, 3): 8,
        (3, 1, 1, 2): 9
    }

    _LEFT_EDGES_TO_CODE_MAPPING = {
        (1, 1, 2, 3): "G",
        (1, 2, 2, 2): "G",
        (2, 2, 1, 2): "G",
        (1, 1, 4, 1): "G",
        (2, 3, 1, 1): "G",
        (1, 3, 2, 1): "G",
        (4, 1, 1, 1): "G",
        (2, 1, 3, 1): "G",
        (3, 1, 2, 1): "G",
        (2, 1, 1, 3): "G",

        (3, 2, 1, 1): "L",
        (2, 2, 2, 1): "L",
        (2, 1, 2, 2): "L",
        (1, 4, 1, 1): "L",
        (1, 1, 3, 2): "L",
        (1, 2, 3, 1): "L",
        (1, 1, 1, 4): "L",
        (1, 3, 1, 2): "L",
        (1, 2, 1, 3): "L",
        (3, 1, 1, 2): "L",
    }

    _PARITY_SEQUENCE_MAPPING = {
        ("L", "L", "L", "L", "L", "L"): 0,
        ("L", "L", "G", "L", "G", "G"): 1,
        ("L", "L", "G", "G", "L", "G"): 2,
        ("L", "L", "G", "G", "G", "L"): 3,
        ("L", "G", "L", "L", "G", "G"): 4,
        ("L", "G", "G", "L", "L", "G"): 5,
        ("L", "G", "G", "G", "L", "L"): 6,
        ("L", "G", "L", "G", "L", "G"): 7,
        ("L", "G", "L", "G", "G", "L"): 8,
        ("L", "G", "G", "L", "G", "L"): 9,
    }

    def decode_barcode(self, image: np.ndarray) -> str | None:
        """
        Decodes the barcode from an image
        :param image: The cropped image that shows only the barcode to decode
        :return: The numbers from the barcode as string
        """

        ean_13_parity_digit = 0
        results = []

        # Extract the scanlines
        scanlines = self._extract_scanlines(image)

        for scanline in scanlines:
            # Get the edge signal
            gradient = self._normalize_gradient(scanline)

            # Find the edges
            edges = self._detect_edges(gradient)

            # Estimate the width between the fixed edges
            delta = self._estimate_module_width(edges)

            # Get the fixed edges
            fixed_edges = self._viterbi_edges(gradient, edges, expected_num_edges=60, delta=delta)

            # Get the left digits together with their patterns
            left_digits, pattern = self._viterbi_digit_decoding_left(fixed_edges, delta=delta)

            # Get the right digits
            right_digits = self._viterbi_digit_decoding_right(fixed_edges, delta=delta)

            # Concatenate the left and right digits
            possible_digits = left_digits[0:6] + right_digits

            # Get the parity digit of the ean 13 code
            parity_pattern = tuple(str(p) for p in pattern[0:6])
            ean_13_parity_digit = Decoder._PARITY_SEQUENCE_MAPPING.get(
                parity_pattern) if parity_pattern in Decoder._PARITY_SEQUENCE_MAPPING else 0

            # Add the possible read
            results.append(possible_digits)

        # Create a numpy array
        results = np.array(results)

        # Get the final digits by simple counting the numbers for each index and insert the parity digit
        final_digits = [int(np.bincount(results[:, i]).argmax()) for i in range(12)]
        final_digits.insert(0, ean_13_parity_digit)

        # Verify the checksum
        is_correct = self._verify_checksum(final_digits)

        if is_correct:
            return "".join(str(d) for d in final_digits)

        return None

    def _extract_scanlines(self, image: np.ndarray, num_lines=1) -> list[np.ndarray]:
        """
        Extracts scanlines from the binary image of the barcode
        :param image: The image of the barcode
        :param num_lines: The number of scanlines to extract
        :return: An array of the scanlines
        """

        # Get the dimension of the image and the boundaries of the area to extract scanlines from
        height, width = image.shape
        scan_area_bottom = height * 0.3
        scan_area_top = height * 0.7

        # Get a list of y-coordinates depending on the boundaries and the number of scanlines
        y_coordinates = np.linspace(scan_area_bottom, scan_area_top, num_lines).astype(int)

        return [image[y_coordinate, :] for y_coordinate in y_coordinates]

    def _normalize_gradient(self, scanline: np.ndarray) -> np.ndarray:
        """
        Normalize the gradient of the scan line
        :param scanline: The scan line to normalize
        :return:
        """

        gradient = np.gradient(scanline.astype(float))
        gradient = gradient / (np.max(np.abs(gradient)) + 1e-6)

        return gradient

    def _detect_edges(self, gradient: np.ndarray) -> np.ndarray:
        """
        Detects the edges by finding the peaks
        :param gradient: The gradient of the scan line
        :return: The peaks as array
        """

        peaks_max, _ = find_peaks(gradient)
        peaks_min, _ = find_peaks(-gradient)
        peaks = np.sort(np.concatenate([peaks_max, peaks_min]))

        return peaks

    def _estimate_module_width(self, edges: np.ndarray) -> float:
        """
        Estimates the delta between the edges of the barcode
        :param edges: The array of edges
        :return: The estimated delta between the barcodes
        """

        return (edges[-1] - edges[0]) / 95.0

    def _viterbi_edges(self, gradient: np.ndarray, edge_candidates: np.ndarray, expected_num_edges=60,
                       delta=None) -> np.ndarray:
        """
        Finds the optimal edges for a typical barcode pattern
        :param gradient: The edge signal to search on
        :param edge_candidates: The possible edge candidates
        :param expected_num_edges: The number of expected edges
        :param delta: The calculates distance between the edges
        :return: The optimal edges of the barcode
        """

        edge_candidates_num = len(edge_candidates)
        cost = np.full((edge_candidates_num, expected_num_edges), np.inf)
        backpointer = np.full((edge_candidates_num, expected_num_edges), -1, dtype=int)

        cost[:, 0] = -np.abs(gradient[edge_candidates])

        for n in range(1, expected_num_edges):
            for j in range(edge_candidates_num):
                for i in range(j):
                    spacing_penalty = 0
                    if delta:
                        expected_spacing = delta
                        spacing = edge_candidates[j] - edge_candidates[i]
                        spacing_penalty = ((spacing - expected_spacing) ** 2)
                    candidate_cost = cost[i, n - 1] - np.abs(gradient[edge_candidates[j]]) + 0.01 * spacing_penalty
                    if candidate_cost < cost[j, n]:
                        cost[j, n] = candidate_cost
                        backpointer[j, n] = i

        best_end = np.argmin(cost[:, expected_num_edges - 1])
        path = []
        idx = best_end
        for n in reversed(range(expected_num_edges)):
            path.append(edge_candidates[idx])
            idx = backpointer[idx, n]

        return np.array(path[::-1])

    def _viterbi_digit_decoding_left(self, edge_positions: np.ndarray, delta: float):
        """
        Decodes the digit candidates
        :param edge_positions: The edge positions of the scan line
        :param delta: The distance between the edges
        :return: The digit candidates
        """

        digit_candidates = []
        start = 3

        for i in range(12):
            base = start + i * 4

            widths = np.diff(edge_positions[base:base + 5]) / delta
            candidates = self._generate_digit_candidates_left(widths)
            digit_candidates.append(candidates)

        T = len(digit_candidates)

        states = []
        for t in range(T):
            for (digit, error, pattern) in digit_candidates[t]:
                states.append((t, digit, pattern))

        cost = {}
        path = {}

        for i, (t, digit, pattern) in enumerate(states):
            if t == 0:
                error = next(e for d, e, p in digit_candidates[0] if d == digit and p == pattern)
                cost[(t, digit, pattern)] = error
                path[(t, digit, pattern)] = None

        for t in range(1, T):
            for d_curr, err_curr, p_curr in digit_candidates[t]:
                for d_prev, err_prev, p_prev in digit_candidates[t - 1]:
                    prev_key = (t - 1, d_prev, p_prev)
                    if prev_key in cost:
                        c = cost[prev_key] + err_curr
                        curr_key = (t, d_curr, p_curr)
                        if c < cost.get(curr_key, np.inf):
                            cost[curr_key] = c
                            path[curr_key] = prev_key

        final_states = [k for k in cost.keys() if k[0] == T - 1]
        final_state = min(final_states, key=lambda k: cost[k])

        sequence = []
        pattern_sequence = []
        state = final_state
        while state is not None:
            t, digit, pattern = state
            sequence.append(digit)
            pattern_sequence.append(pattern)
            state = path[state]

        sequence.reverse()
        pattern_sequence.reverse()

        patterns = []
        for pattern in pattern_sequence:
            patterns.append(Decoder._LEFT_EDGES_TO_CODE_MAPPING.get(pattern))

        return list(sequence), list(patterns)

    def _viterbi_digit_decoding_right(self, edge_positions: np.ndarray, delta: float):
        """
        Decodes the digit candidates from the right side
        :param edge_positions: The positions of the edges
        :param delta: The delta
        :return: A list of digits candidates
        """

        digit_candidates = []
        start = 32

        for i in range(6):
            base = start + i * 4
            widths = np.diff(edge_positions[base:base + 5]) / delta
            candidates = self._generate_digit_candidates_right(widths)
            digit_candidates.append(candidates)

        T = len(digit_candidates)
        D = 10

        cost = np.full((T, D), np.inf)
        path = np.full((T, D), -1, dtype=int)

        for d, err in digit_candidates[0]:
            cost[0, d] = err

        for t in range(1, T):
            for d_curr, err_curr in digit_candidates[t]:
                for d_prev in range(D):
                    c = cost[t - 1, d_prev] + err_curr
                    if c < cost[t, d_curr]:
                        cost[t, d_curr] = c
                        path[t, d_curr] = d_prev

        # Backtrack
        final_digit = np.argmin(cost[-1])
        sequence = [final_digit]
        for t in reversed(range(1, T)):
            final_digit = path[t, final_digit]
            sequence.append(final_digit)

        return list(reversed(sequence))

    def _generate_digit_candidates_left(self, widths: np.ndarray) -> list:
        """
        Generates the digit candidates between the start guard and the middle guard
        :param widths: The edge widths
        :return: The digit candidates
        """

        candidates = []
        for pattern, digit in Decoder._LEFT_EDGES_TO_DIGIT_MAPPING.items():
            error = np.sum((np.array(widths) - np.array(pattern)) ** 2)
            candidates.append((digit, error, pattern))

        return candidates

    def _generate_digit_candidates_right(self, widths: np.ndarray) -> list:
        """
        Generates the digit candidates between the end guard and the middle guard
        :param widths: The edge widths
        :return: The digit candidates
        """

        candidates = []
        for pattern, digit in Decoder._RIGHT_EDGES_TO_DIGIT_MAPPING.items():
            error = np.sum((np.array(widths) - np.array(pattern)) ** 2)
            candidates.append((digit, error))

        return candidates

    def _verify_checksum(self, digits: list) -> bool:
        """
        Verifies the digit sequence by calculating the checksum
        :param digits:
        :return: True if the digit sequence is correct, otherwise false
        """

        sum_odd_numbers = sum(digits[i] for i in range(1, 12, 2))
        sum_even_numbers = sum(digits[i] for i in range(0, 12, 2))
        checksum = (10 - ((3 * sum_odd_numbers + sum_even_numbers) % 10)) % 10

        return checksum == digits[-1]
