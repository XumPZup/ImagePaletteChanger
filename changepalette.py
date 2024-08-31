from itertools import permutations
from collections import Counter
from random import randint, choice
import numpy as np
from argparse import ArgumentParser
import cv2

# Util for getting the command parameters
def get_parameters():
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Input image file', required=True)
    parser.add_argument('-s', '--palette-size', type=int, help='Size of the palette')
    parser.add_argument('-p', '--palette', type=int, nargs='+', help='Palette')
    parser.add_argument('-o', '--output', type=str, help='Output file name')

    args = parser.parse_args()
    if not args.palette is None:
        # Return arguments with random permutation of the palette
        return args.image, len(args.palette), choice(list(permutations(args.palette))), args.output
    return args.image, args.palette_size, args.palette, args.output
        

# Function to map hue to a smaller palette
def reduce_hues(hue_channel, num_colors):
    # Quantize the hue values to the nearest palette value
    max_hue = 180
    interval = max_hue // num_colors
    reduced_hues = (hue_channel // interval) * interval
    return reduced_hues


def apply_palette(hue_channel, palette_mapping):
    # Map the hues to the new palette
    for original, new in palette_mapping:
        hue_channel[hue_channel == original] = new
    return hue_channel

# Remaps the colors using distance from the original
def remap(hue_channel_original, hue_channel_reduced, hue_channel_new):
    return hue_channel_new - (hue_channel_original - hue_channel_reduced)


# Step 0: Get parameters
file_name, num_colors, user_palette, output_file = get_parameters()
print(f"Input file: {file_name}\nPalette size: {num_colors}")
if user_palette is None:
    print("Palette: random")
else:
    print(f"Palette: {user_palette}")

# Step 1: Load the image
img = cv2.imread(file_name)

# Step 2: Change the image to HSV
image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(image_hsv)

# Step 3: Extract the unique values of hue
unique_hues = np.unique(h)

# Step 4: Reduce the representation of colors
h_reduced = reduce_hues(h, num_colors)

# Step 5: Apply the new colors to the image
image_hsv_reduced = cv2.merge([h_reduced, s, v])
image_reduced = cv2.cvtColor(image_hsv_reduced, cv2.COLOR_HSV2BGR)

# Step 6: Find the n most popular colors
# n_colors = 5
h_flat = h_reduced.flatten()
color_counts = Counter(h_flat)
most_common_hues = color_counts.most_common(num_colors) # Not neaded since I'm using the same colors as in `h_reduced`

# Step 7: Apply a palette to the image
if user_palette is None: # Randomize palette if there is none
    color_mapping = [(i[0], randint(0, 180)) for i in most_common_hues]
else:
    color_mapping = [(i[0], new_color) for i, new_color in zip(most_common_hues, user_palette)]

h_palette_applied = apply_palette(h_reduced.copy(), color_mapping)
image_hsv_palette = cv2.merge([h_palette_applied, s, v]) # Image

# Step 7.1 Remap the new palette to the original image
h_palette_remap = remap(h, h_reduced, h_palette_applied)
image_hsv_palette_remap = cv2.merge([h_palette_remap, s, v])

# Step 8: Using the original image, map the palette
image_palette = cv2.cvtColor(image_hsv_palette, cv2.COLOR_HSV2BGR)
image_palette_remap = cv2.cvtColor(image_hsv_palette_remap, cv2.COLOR_HSV2BGR)

# Save results
if output_file is None:
    extension = file_name.split('.')[1]
    output_file = 'output.' + extension
cv2.imwrite('reduced_img.jpg', image_reduced)
cv2.imwrite(f'reduced_{output_file}', image_palette)
cv2.imwrite(output_file, image_palette_remap)
