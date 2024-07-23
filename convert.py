import os
from PIL import Image
import numpy as np
import cv2
import argparse

def convert_to_grayscale_l_channel(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            rgb_image = image.convert('RGB')

            rgb_image_np = np.array(rgb_image)

            lab_image = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2LAB)

            L_channel = lab_image[:, :, 0]

            L_channel_pil = Image.fromarray(L_channel)

            output_path = os.path.join(output_folder, filename)
            L_channel_pil.save(output_path)

            print(f'Converted {filename} to grayscale (L channel) and saved to {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert images to grayscale L channel in LAB color space.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing images.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where grayscale images will be saved.')

    args = parser.parse_args()

    convert_to_grayscale_l_channel(args.input_folder, args.output_folder)
