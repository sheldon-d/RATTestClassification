import cv2
import numpy as np

def convertToGrayscaleAndBlur(pixel_array, kernel_size=(7, 7)):
    # Convert pixel array to grayscale
    px_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)

    # Stretch values to lie between 0 and 255
    px_array = cv2.convertScaleAbs(px_array, alpha=1, beta=0)

    # Blur pixel array using kernel
    px_array = cv2.GaussianBlur(px_array, kernel_size, 0)

    return px_array


def computeMorphologicalClosing(pixel_array, num_closing_steps, kernel_size=(5, 5)):
    # Performing morphological closing using several dilation and erosion steps
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    px_array = pixel_array
    for n in range(num_closing_steps):
        px_array = cv2.dilate(px_array, morph_kernel)
    for n in range(num_closing_steps):
        px_array = cv2.erode(px_array, morph_kernel)

    return px_array


def findBoundingRect(pixel_array, contours, ratio, max_proportion=1.0, draw_all_contours=False):
    rows, cols, layers = pixel_array.shape
    max_area = max_proportion * (rows * cols)
    px_contours = pixel_array.copy()

    if draw_all_contours:
        cv2.drawContours(px_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)

    # Width to height ratio is calculated by inverting ratio if more rows
    if rows >= cols:
        ratio = 1/ratio

    largest = getLargestContour(contours, ratio=ratio, max_area=max_area)
    x, y, w, h = cv2.boundingRect(largest)
    cv2.rectangle(px_contours, (x, y), (x + w, y + h), (0, 0, 255), 2)
    px_cropped = pixel_array[y:y + h, x:x + w]

    return px_contours, px_cropped


def getLargestContour(contours, ratio, max_area):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    max_ratio = 5
    min_ratio = ratio

    # Ensures that width to height ratio is correct depending on orientation of test
    if ratio < 1:
        min_ratio = 1/max_ratio
        max_ratio = ratio

    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        r = w / h
        if min_ratio < r < max_ratio and (cv2.contourArea(contour) <= max_area):
            return contour
    return sorted_contours[0]


def getComponentColours(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convert to coloured components for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Set background label to black
    labeled_img[label_hue == 0] = 0

    return labeled_img


def extractTest(pixel_array):
    px_array = convertToGrayscaleAndBlur(pixel_array)

    # Defining number of steps for morphological closing
    num_closing_steps = 7
    rows, cols, layers = pixel_array.shape
    min_size = 300

    if rows < min_size or cols < min_size:
        num_closing_steps = 1

    # Perform Canny edge detection
    px_array = cv2.Canny(px_array, 30, 70)
    px_edges = px_array

    # Performing morphological closing
    px_array = computeMorphologicalClosing(px_array, num_closing_steps=num_closing_steps)
    px_morph_closing = px_array

    contours, hierarchy = cv2.findContours(px_array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    px_contours, px_cropped = findBoundingRect(pixel_array, contours, ratio=2)

    return px_edges, px_morph_closing, px_contours, px_cropped


def extractIndicator(pixel_array):
    px_array = convertToGrayscaleAndBlur(pixel_array)
    rows, cols, layers = pixel_array.shape
    min_size = 60

    # Defining kernel block size for adaptive thresholding
    block_size = 7

    # Defining number of steps for morphological closing
    num_closing_steps = 1

    if rows < min_size:
        block_size = 5
        num_closing_steps = 0
    if cols < min_size:
        block_size = 3
        num_closing_steps = 0

    # Perform adaptive thresholding
    px_array = cv2.adaptiveThreshold(px_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, 2)

    # Performing morphological closing
    px_array = computeMorphologicalClosing(px_array, num_closing_steps=num_closing_steps)
    px_morph_closing = px_array

    contours, hierarchy = cv2.findContours(px_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    px_contours, px_cropped = findBoundingRect(pixel_array, contours, ratio=3, max_proportion=0.2)

    return px_morph_closing, px_contours, px_cropped


def processIndicator(pixel_array):
    rows, cols, layers = pixel_array.shape

    # Crop image to only include area comprising the indicator lines
    if rows > cols:
        rows_cropped = 0.25
        cols_cropped = 0.4
    else:
        rows_cropped = 0.4
        cols_cropped = 0.25

    start_row = round(rows_cropped * rows)
    end_row = round((1-rows_cropped) * rows)
    start_col = round(cols_cropped * cols)
    end_col = round((1 - cols_cropped) * cols)

    px_array = pixel_array[start_row:end_row, start_col:end_col]
    px_array = convertToGrayscaleAndBlur(px_array)

    # Compress colors of image into a narrower range, decreasing contrast
    px_array = cv2.convertScaleAbs(px_array, alpha=0.1, beta=0)
    ret, px_array = cv2.threshold(px_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all connected components in image (including the background)
    num_labels, labels = cv2.connectedComponents(px_array)

    # Colour each indicator line differently
    labeled_img = getComponentColours(labels)

    # Subtract one to exclude background from being a connected component
    num_lines = num_labels - 1

    return labeled_img, num_lines


def processResult(num_lines):
    if num_lines == 2:
        return "Positive"
    elif num_lines == 1:
        return "Negative"
    else:
        return "Invalid"
