import os
import cv2


def gaussian_filter(input_dir: str, output_dir: str) -> None:
    """
        Apply different Gaussian filters to each image in the input directory,
        and save them in the output directory

        Parameters:
        - input_dir (str): Path to the input directory containing the images
        - output_dir (str): Path to the output directory, where the filtered images will be saved
    """
    # create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # define gaussian filter parameters
    filter_params = [
        {"kernel_size": (3,3), "sigma": 0.3},
        {"kernel_size": (7,7), "sigma": 1},
        {"kernel_size": (11,11), "sigma": 1.6}
    ]

    # iterate through folders in the input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        # ensure it is a directory because files can also be present
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_dir, folder_name)

            # create output folder if it doesn't exist
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # iterate through images in the folder
            for file_name in os.listdir(folder_path):

                # check if the file is an image
                if file_name.endswith(('jpg', 'jpeg', 'png')):

                    # load the image
                    file_path = os.path.join(folder_path, file_name)
                    image = cv2.imread(file_path)
                    
                    # apply different gaussian filters
                    for params in filter_params:
                        kernel_size = params["kernel_size"]
                        sigma = params["sigma"]
                        filtered_image = cv2.GaussianBlur(image, kernel_size, sigma)

                        # save the filtered image
                        output_image_path = os.path.join(output_folder_path, f"{file_name.split('.')[0]}_kernel_{kernel_size[1]}_sigma_{sigma}.png")
                        cv2.imwrite(output_image_path, filtered_image)
                        