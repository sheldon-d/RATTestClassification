import sys
from pathlib import Path

from ImageProcessingFunctions import *
from matplotlib import pyplot
from natsort import natsorted


def main():
    command_line_arguments = sys.argv[1:]
    output_path = Path("output_images")
    SHOW_DEBUG_FIGURES = False

    if len(command_line_arguments) > 0:
        input_filename = command_line_arguments[0]
        input_files = [input_filename]
        SHOW_DEBUG_FIGURES = True
    else:
        input_path = Path("images")
        input_files = natsorted([str(p) for p in input_path.iterdir()])

    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    getResultFromImages(input_files, output_path, intermediate_figs=SHOW_DEBUG_FIGURES)


def getResultFromImages(input_files, output_path, intermediate_figs):
    # Assuming same number of test samples in each class (Pos/Neg)
    num_files = len(input_files)
    class_num = round(num_files / 2)

    # Set up the plots for summary results in a figure
    if num_files > 1:
        fig_pos, axs_pos = pyplot.subplots(2, class_num)
        fig_neg, axs_neg = pyplot.subplots(2, class_num)

        fig_pos.suptitle(f"Positive COVID-19 RATs summary", fontsize=20, fontweight="bold")
        fig_neg.suptitle(f"Negative COVID-19 RATs summary", fontsize=20, fontweight="bold")
    else:
        axs_pos, axs_neg = None, None

    for i in range(len(input_files)):
        input_filename = input_files[i]
        file_path = Path(input_filename)
        extension = file_path.suffix
        output_filename = output_path / file_path.name.replace(extension, "_output.png")

        # Read image in grayscale mode
        pixel_array = cv2.imread(input_filename)

        # Processing image to find region where RAT test is placed
        px_edges, px_morph_closing, px_contours, px_cropped = extractTest(pixel_array)

        # Finding region where indicators are present
        indicator_closing, indicator_contours, indicator_cropped = extractIndicator(px_cropped)

        # Get number of indicator lines and print result
        indicator_lines, num_lines = processIndicator(indicator_cropped)
        result = processResult(num_lines)
        prefix = f"{i + 1}. " if num_files > 1 else ""
        print(f"{prefix}Result: {result}")

        # Write the output image into output_filename
        cv2.imwrite(str(output_filename), indicator_lines)

        if intermediate_figs:
            # Set up the plots for intermediate results in a figure
            fig1, axs1 = pyplot.subplots(2, 4)
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
            axs1[1, 0].imshow(indicator_closing, cmap='gray')
            axs1[1, 1].set_title('Bounding box for indicator')
            axs1[1, 1].imshow(cv2.cvtColor(indicator_contours, cv2.COLOR_BGR2RGB))
            axs1[1, 2].set_title('Extracted region of indicator')
            axs1[1, 2].imshow(cv2.cvtColor(indicator_cropped, cv2.COLOR_BGR2RGB))
            axs1[1, 3].set_title('Extracted lines using CCA')
            axs1[1, 3].imshow(cv2.cvtColor(indicator_lines, cv2.COLOR_BGR2RGB))

        if num_files > 1:
            if i < class_num:
                axs_pos[0, i % class_num].set_title(f"Test {i + 1}")
                axs_pos[0, i % class_num].imshow(cv2.cvtColor(indicator_contours, cv2.COLOR_BGR2RGB))
                axs_pos[1, i % class_num].set_title(f"Result {i + 1}")
                axs_pos[1, i % class_num].imshow(cv2.cvtColor(indicator_lines, cv2.COLOR_BGR2RGB))
            else:
                axs_neg[0, i % class_num].set_title(f"Test {i + 1}")
                axs_neg[0, i % class_num].imshow(cv2.cvtColor(indicator_contours, cv2.COLOR_BGR2RGB))
                axs_neg[1, i % class_num].set_title(f"Result {i + 1}")
                axs_neg[1, i % class_num].imshow(cv2.cvtColor(indicator_lines, cv2.COLOR_BGR2RGB))

    if intermediate_figs or num_files > 1:
        # Plot the current figures
        pyplot.show()


if __name__ == "__main__":
    main()
