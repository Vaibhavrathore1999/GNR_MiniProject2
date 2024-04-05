import os
import cv2


def downscale_images(input_dir: str, output_dir: str, target_width: int, target_height: int) -> None:
    """
        Downscale images in the input directory to the specified dimensions,
        save them in the output directory

        Parameters:
        - input_dir (str): Path to the input directory containing the images
        - output_dir (str): Path to the output directory where downscaled images will be saved
        - target_width (int): Target width for downscaled images
        - target_height (int): Target height for downscaled images
    """

    # create ouput directory if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # iterate through folders in input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        # ensure it is a directory (because files can also be present)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_dir, folder_name)

            # create output folder if doesn't exist
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # iterate through images in the folder
            for file_name in os.listdir(folder_path):

                # check if the file is an image
                if file_name.endswith(('jpg', 'jpeg', 'png')):
                    
                    # load the image
                    file_path = os.path.join(folder_path, file_name)
                    image = cv2.imread(file_path)

                    # downscale the image
                    downscaled_image = cv2.resize(image, (target_width, target_height))                            # resize to (width, height)

                    # save the downscaled image
                    output_image_path = os.path.join(output_folder_path, file_name)
                    cv2.imwrite(output_image_path, downscaled_image)
                    