"""
Steps :
1. Read the Test image and convert it to grayscale using G = round(0.299R + 0.587G + 0.114B)
2. Calculate the Gradient Magnitude using Prewitt's operator. Normalize gradient values and then compute the Gradient angle.
3. Compute the descriptor for test image which will be of dimension 7524 x 1.
4. Perform training. Repeat steps 1 to 3 for all 20 training images.
5. Calculate similarity by using Histogram Intersection. Find the 3 nearest neighbors.
6. Classify the image as Human or No-human
"""
import numpy as np
import argparse
import cv2
import os


def convolution(image, mask):
    image_row, image_col = image.shape
    mask_row, mask_col = mask.shape
    convoluted_image = np.zeros(image.shape)
    add_row = int(mask_row - 1) // 2
    add_col = int(mask_col - 1) // 2

    # Initialising a 2D array with zeroes along with extra rows and columns to handle the undefined values
    modified_image = np.zeros((image_row + (2 * add_row), image_col + (2 * add_col)))

    modified_image_row, modified_image_col = modified_image.shape

    # Defining the region of interest for the input image
    modified_image[add_row: modified_image_row - add_row, add_col:modified_image_col - add_col] = image

    # Matrix multiplication - Performing convolution on image with kernel
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            # Using sliding window concept for matrix multiplication of kernel and region of interest of input image
            convoluted_image[row, col] = np.sum(mask * modified_image[row: row + mask_row, col: col + mask_col])

    return convoluted_image


def gradient_operation(image, edge_filter):
    # Computing the horizontal gradient by performing convolution the input image with Prewitt's horizontal edge filter
    horizontal_gradient = convolution(image, edge_filter)
    # Output: [[1,1,1], [0,0,0], [-1,-1,-1]]
    vertical_edge_filter = np.flip(edge_filter.T, axis=0)

    # Computing the vertical gradient by performing convolution the input image with Prewitt's vertical edge filter
    vertical_gradient = convolution(image, vertical_edge_filter)

    # Using the formula, gradient magnitude = Square Root of Squares of Horizontal and Vertical Gradient
    gradient_magnitude = np.sqrt(np.square(horizontal_gradient) + np.square(vertical_gradient))
    gradient_magnitude = gradient_magnitude / (3 * (np.sqrt(2)))



# Calculating gradient angle -> tan inverse (vertical gradient/horizontal gradient) in radians
    gradient_angle = np.arctan2(vertical_gradient, horizontal_gradient)

    # Converting gradient angle from radians to degree which returns in the range of -180 to 180.
    gradient_direction = np.rad2deg(gradient_angle)

    # If angle is negative add 360 to make it positive
    gradient_direction[gradient_direction < 0] += 360

    # If angle is greater than 180, subtract 180
    gradient_direction[gradient_direction > 180] -= 180

    return horizontal_gradient, vertical_gradient, gradient_magnitude, gradient_direction


def histogram_calculate(pixel_mag, pixel_angle, orientation_bin_midpoints, cell_list):

    # If the gradient angle for a pixel is between 10 and 170 then calculate bin centres and corresponding indexes
    # for the histogram
    if 10 <= pixel_angle < 170:
        for i in range(len(orientation_bin_midpoints)):
            if pixel_angle < orientation_bin_midpoints[i]:
                bin1 = orientation_bin_midpoints[i - 1]
                idx1 = i - 1
                idx2 = i
                ratio1 = abs(pixel_angle - bin1) / 20;
                ratio2 = abs(20 - (pixel_angle - bin1)) / 20;

                break
    elif 0 <= pixel_angle < 10:
        bin2 = 10
        idx1 = 0
        idx2 = len(orientation_bin_midpoints) - 1
        ratio1 = (bin2 - pixel_angle) / 20;
        ratio2 = (20 - (bin2 - pixel_angle)) / 20;
    else:
        bin1 = 170
        idx1 = len(orientation_bin_midpoints) - 1
        idx2 = 0
        ratio1 = abs(pixel_angle - bin1) / 20
        ratio2 = abs(20 - (pixel_angle - bin1)) / 20

    # Updating the 2 bins in the histogram depending on the distance of pixel angle from the bin centres
    cell_list[idx1] += pixel_mag * ratio2
    cell_list[idx2] += pixel_mag * ratio1


def histogram_cell(gradient_magnitude, gradient_direction, orientation_bin_midpoints, output_block):

    # Calculating histogram for each cell and saving the magnitudes in the cell_list array which has length 9.
    cell_list = np.zeros(9, dtype=float)
    for i in range(gradient_magnitude.shape[0]):
        for j in range(gradient_magnitude.shape[1]):
            histogram_calculate(gradient_magnitude[i][j], gradient_direction[i][j], orientation_bin_midpoints,
                                cell_list)
    # Concatenating the histogram cells into one array of dimension 1 x 36.
    output_block.extend(list(cell_list))

def histogram_block(gradient_magnitude, gradient_direction, orientation_bin_midpoints, descriptor):
    output_block = []

    # Going through all 4 cells in a block to compute histogram. Dimension of a cell is 8 x 8 pixels.
    for i in range(0, gradient_magnitude.shape[0], 8):
        for j in range(0, gradient_magnitude.shape[1], 8):
            histogram_cell(gradient_magnitude[i:i + 8, j:j + 8], gradient_direction[i:i + 8, j:j + 8],
                           orientation_bin_midpoints, output_block)

    # Getting the L2-Norm value for each block
    normalized_block_value = get_l2_norm(np.array(output_block))

    # Dividing all the values by L2-Norm value
    if normalized_block_value != 0:
        output_block = output_block / normalized_block_value
    descriptor.extend(output_block)


def histogram_image(gradient_magnitude, gradient_direction, orientation_bin_midpoints, descriptor):
    # Creating overlapping blocks of the image. Dimension of the block is 16 x 16 pixels or 2 x 2 cells.
    for i in range(0, gradient_magnitude.shape[0] - 8, 8):
        for j in range(0, gradient_magnitude.shape[1] - 8, 8):
            histogram_block(gradient_magnitude[i:i + 16, j:j + 16], gradient_direction[i:i + 16, j:j + 16],
                            orientation_bin_midpoints, descriptor)

def get_l2_norm(block):
    l2_norm = 0
    for i in range(block.shape[0]):
        # Summing the squares of each block element
        l2_norm += np.array(block[i]) ** 2

    # Returning squareroot of the sum
    return np.sqrt(l2_norm)

def histogram_intersection(train_img, test_img):
    # Applying the histogram intersection formula to calculate similarity
    total = np.sum(train_img)
    min_dist = 0
    for i in range(0, train_img.shape[0]):
        min_dist += min(train_img[i], test_img[i])
    return min_dist / total


def knn(neighbour_info, training_set):
    # To find the majority of the 3 nearest neighbors to the test image
    print("The 3 nearest neighbors are : ")
    majority = 0
    for k, v in neighbour_info.items():
        if training_set.get(k) == "Human":
            majority += 1
        print(k, "\t", v, "\t", training_set.get(k))
    if majority > 1:
        print("Human Detected!")
    else:
        print("Human Not Detected!")


if __name__ == '__main__':
    print("Running Human Detection using HOG Feature!")

    # Reading the input file name from the arguments passed from command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    frame = cv2.imread(args['image'])

    # Converting the image to grayscale
    blue, green, red = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    test_image = np.around(0.299 * red + 0.587 * green + 0.114 * blue)

    # Declaring the Prewitt's operator
    edge_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='int')

    # Computing gradient magnitude and gradient angle for test image
    horizontal_gradient, vertical_gradient, gradient_magnitude, gradient_direction = gradient_operation(test_image,
                                                                                                        edge_filter)
    # Creating path to write output images of Canny Edge Detector
    folder, fname_with_extension = os.path.split(args['image'])
    fname, extension = os.path.splitext(fname_with_extension)
    path = str(fname) + "_output"
    access = 0o755
    cv2.imwrite(fname + "_GradientMagnitude.bmp", gradient_magnitude)

    descriptor = []

    # Calculating bin centers for the histogram. Number of bins = 9.
    orientations = 9
    orientations_arr = np.arange(orientations)
    orientation_bin_midpoints = (
            180 * (orientations_arr + .5) / orientations)

    similarity = {}
    # Hard coding the training images with their equivalent labels.
    training_dataset = {"01-03e_cut": "No-human", "00000053a_cut": "No-human", "00000057a_cut": "No-human",
                        "00000062a_cut": "No-human",
                        "00000091a_cut": "No-human", "00000093a_cut": "No-human",
                        "no_person__no_bike_213_cut": "No-human", "no_person__no_bike_219_cut": "No-human",
                        "no_person__no_bike_247_cut": "No-human", "no_person__no_bike_259_cut": "No-human",
                        "crop001008b": "Human"
                        , "crop001028a": "Human", "crop001030c": "Human", "crop001045b": "Human", "crop001047b": "Human",
                        "crop001063b": "Human", "crop001275b": "Human", "crop001672b": "Human",
                        "person_and_bike_026a": "Human"
                        }
    # Computing descriptor for the test image
    histogram_image(gradient_magnitude, gradient_direction, orientation_bin_midpoints, descriptor)
    descriptor_array = np.array(descriptor)

    # Saving the descriptor of the test image to an output file.
    file = open(fname + "_descriptor.txt", "w+")
    for output in descriptor_array:
        content = str(output) + "\n"
        file.write(content)
    file.close()

    # Running a loop through all the training images to compute a descriptor for each.
    for key, value in training_dataset.items():
        # Reading all the images from a directory named "training".
        frame_train = cv2.imread("training/" + key + ".bmp")

        # Converting every training image to grayscale.
        train_image = np.round(
            0.299 * frame_train[:, :, 0] + 0.587 * frame_train[:, :, 1] + 0.114 * frame_train[:, :, 2], decimals=5)

        # Computing gradient magnitudes and gradient angles for every training image.
        horizontal_gradient1, vertical_gradient1, gradient_magnitude1, gradient_direction1 = gradient_operation(
            train_image,
            edge_filter)

        # Computing descriptor for every training image
        training_image_descriptor = []
        histogram_image(gradient_magnitude1, gradient_direction1, orientation_bin_midpoints, training_image_descriptor)
        train_image_array = np.array(training_image_descriptor)

        # Calculating the similarity of the test image with every training image.
        s = histogram_intersection(train_image_array, descriptor_array)
        similarity[key] = s

    similarity = {k: v for k, v in sorted(similarity.items(), key=lambda item: item[1], reverse=True)}
    # Choose first 3 with highest similarity from list sorted in non-increasing order
    nearest_3 = dict(list(similarity.items())[0: 3])

    # Predicting if image is Human or No-Human.
    knn(nearest_3, training_dataset)
