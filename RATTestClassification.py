import sys
from pathlib import Path

from ImageProcessingFunctions import *
from matplotlib import pyplot


def main():
    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = ".\images\image1.png"

    if len(command_line_arguments) > 0:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = True

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(input_filename).name.replace(".png", "_output.png")

    # Read image in grayscale mode
    pixel_array = cv2.imread(input_filename)

    # Set up the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 4)

    # Processing image to find region where RAT test is placed
    px_edges, px_morph_closing, px_contours, px_cropped = extractTest(pixel_array)

    # Finding region where indicators are present
    new_closing, new_contours, new_cropped = extractIndicator(px_cropped)

    # Get number of indicator lines and print result
    px_lines, num_lines = processIndicator(new_cropped)
    result = processResult(num_lines)
    print(f"Result: {result}")

    fig1.suptitle(f"{result} COVID-19 RAT test", fontsize=20, fontweight="bold")

    # Draw a bounding box as a rectangle into the input image
    axs1[0, 0].set_title('Canny edge detection of image')
    axs1[0, 0].imshow(px_edges, cmap='gray')
    axs1[0, 1].set_title(f'Morphological closing of image')
    axs1[0, 1].imshow(px_morph_closing, cmap='gray')
    axs1[0, 2].set_title('Bounding box for test')
    axs1[0, 2].imshow(cv2.cvtColor(px_contours, cv2.COLOR_BGR2RGB))
    axs1[0, 3].set_title('Extracted region of test')
    axs1[0, 3].imshow(cv2.cvtColor(px_cropped, cv2.COLOR_BGR2RGB))

    axs1[1, 0].set_title('Morphological closing of test region')
    axs1[1, 0].imshow(new_closing, cmap='gray')
    axs1[1, 1].set_title('Bounding box for indicator')
    axs1[1, 1].imshow(cv2.cvtColor(new_contours, cv2.COLOR_BGR2RGB))
    axs1[1, 2].set_title('Extracted region of test')
    axs1[1, 2].imshow(cv2.cvtColor(new_cropped, cv2.COLOR_BGR2RGB))
    axs1[1, 3].set_title('Extracted lines using CCA')
    axs1[1, 3].imshow(cv2.cvtColor(px_lines, cv2.COLOR_BGR2RGB))

    # Write the output image into output_filename
    cv2.imwrite(str(output_filename), px_lines)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()