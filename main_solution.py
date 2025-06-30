import cv2
import numpy as np
from scipy.signal import find_peaks

EAN13_LEFT = {
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

EAN13_LEFT_CODE = {
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

PARITY_SEQUENCE_MAPPING = {
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

EAN13_RIGHT = {
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

def normalize_gradient(I):  # works
    e = np.gradient(I.astype(float))
    e = e / (np.max(np.abs(e)) + 1e-6)
    return e

def detect_edges(e):  # works
    peaks_max, _ = find_peaks(e)
    peaks_min, _ = find_peaks(-e)
    peaks = np.sort(np.concatenate([peaks_max, peaks_min]))
    return peaks

def estimate_module_width(edges):  # works
    if len(edges) < 60:
        raise ValueError("Not enough edges")
    return (edges[-1] - edges[0]) / 95.0

def viterbi_edges(e, edges, expected_num_edges=60, delta=None):
    N = len(edges)
    cost = np.full((N, expected_num_edges), np.inf)
    backpointer = np.full((N, expected_num_edges), -1, dtype=int)

    cost[:, 0] = -np.abs(e[edges])  # favor strong edges

    for n in range(1, expected_num_edges):
        for j in range(N):
            for i in range(j):
                spacing_penalty = 0
                if delta:
                    expected_spacing = delta
                    spacing = edges[j] - edges[i]
                    spacing_penalty = ((spacing - expected_spacing) ** 2)
                candidate_cost = cost[i, n - 1] - np.abs(e[edges[j]]) + 0.01 * spacing_penalty
                if candidate_cost < cost[j, n]:
                    cost[j, n] = candidate_cost
                    backpointer[j, n] = i

    best_end = np.argmin(cost[:, expected_num_edges - 1])
    path = []
    idx = best_end
    for n in reversed(range(expected_num_edges)):
        path.append(edges[idx])
        idx = backpointer[idx, n]
    return np.array(path[::-1])


def generate_digit_candidates_left(widths, delta):
    candidates = []
    for pattern, digit in EAN13_LEFT.items():
        error = np.sum((np.array(widths) - np.array(pattern)) ** 2)
        candidates.append((digit, error, pattern))
        # pattern added for code type tracking
    return candidates


def generate_digit_candidates_right(widths, delta):
    candidates = []
    for pattern, digit in EAN13_RIGHT.items():
        error = np.sum((np.array(widths) - np.array(pattern)) ** 2)
        candidates.append((digit, error))
    return candidates


def viterbi_digit_decoding_left(edge_positions, delta):
    digit_candidates = []
    start = 3  # Skip left guard (3 bars) 3
    pattern_sequence = []

    for i in range(12):
        base = start + i * 4

        widths = np.diff(edge_positions[base:base + 5]) / delta
        candidates = generate_digit_candidates_left(widths, delta)
        digit_candidates.append(candidates)

    T = len(digit_candidates)
    D = 10  # Digits 0–9

    cost = np.full((T, D), np.inf, dtype=object)
    path = np.full((T, D), -1, dtype=int)
    patterns = np.full((T, D), np.inf, dtype=object)

    for d, err, p in digit_candidates[0]:
        cost[0, d] = err

    for t in range(1, T):
        for d_curr, err_curr, p_curr in digit_candidates[t]:
            for d_prev in range(D):
                c = cost[t - 1, d_prev] + err_curr
                if c < cost[t, d_curr]:
                    cost[t, d_curr] = c
                    path[t, d_curr] = d_prev
                    patterns[t, d_curr] = p_curr

    final_digit = np.argmin(cost[-1])
    sequence = [final_digit]
    for t in reversed(range(1, T)):
        final_digit = path[t, final_digit]
        sequence.append(final_digit)
        final_pattern = patterns[t, final_digit]
        code = EAN13_LEFT_CODE[final_pattern]
        pattern_sequence.append(code)

    return list(reversed(sequence)), list(pattern_sequence)

def viterbi_digit_decoding_right(edge_positions, delta):
    digit_candidates = []
    start = 32  # Skip all including middle guard

    for i in range(6):
        base = start + i * 4
        # if base + 4 > len(edge_positions):
        # break
        widths = np.diff(edge_positions[base:base + 5]) / delta
        candidates = generate_digit_candidates_right(widths, delta)
        digit_candidates.append(candidates)

    T = len(digit_candidates)
    D = 10  # Digits 0–9

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

def verify_checksum(digits):
    odd = sum(digits[i] for i in range(1, 12, 2))
    even = sum(digits[i] for i in range(0, 12, 2))
    check = (10 - ((3 * odd + even) % 10)) % 10
    print(check)

    return check == digits[-1]

def extract_scanlines(binary_image, num_lines=1):  # num = 1 for testing purpose
    h, w = binary_image.shape
    ys = np.linspace(int(h * 0.3), int(h * 0.7), num_lines).astype(int)
    return [binary_image[y, :] for y in ys]

def model1_read_barcode_from_binary_image(binary_image):
    scanlines = extract_scanlines(binary_image)
    results = []
    for I in scanlines:
        e = normalize_gradient(I)

        edges = detect_edges(e)

        delta = estimate_module_width(edges)

        edge_positions = viterbi_edges(e, edges, expected_num_edges=60, delta=delta)

        digits1, patterns1 = viterbi_digit_decoding_left(edge_positions, delta)

        digits2 = viterbi_digit_decoding_right(edge_positions, delta)

        digits_new = digits1[0:6] + digits2

        # Get pattern slice
        final_pattern = tuple(patterns1[0:6])

        parity_digit = PARITY_SEQUENCE_MAPPING[final_pattern]

        # if digits and len(digits) == 12 and verify_checksum(digits): CHECKSUM IS NEXT TASK
        results.append(digits_new)

    results = np.array(results)
    final_digits = [int(np.bincount(results[:, i]).argmax()) for i in range(12)]

    final_digits.insert(0, parity_digit)

    is_correct = verify_checksum(final_digits)

    return final_digits

if __name__ == "__main__":

    img2 = cv2.imread('barcode.jpg', cv2.IMREAD_GRAYSCALE)
    ret, thresh1 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

    try:
        digits = model1_read_barcode_from_binary_image(thresh1)
        print("Detected barcode of class EAN-13: ", digits)
    except Exception as e:
        print("Error while decoding: ", e)
