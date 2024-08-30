from collections import Counter
from random import randint
import numpy as np
import sys
import cv2


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


# Step 1: Load the image
img = cv2.imread(sys.argv[1])

# Step 2: Change the image to HSV
image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(image_hsv)

# Step 3: Extract the unique values of hue
unique_hues = np.unique(h)

# Step 4: Reduce the representation of colors
try:
    num_colors = int(sys.argv[2])
except IndexError:
    num_colors = 5

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
color_mapping = [(i[0], randint(0, 180)) for i in most_common_hues]
h_palette_applied = apply_palette(h_reduced.copy(), color_mapping)
image_hsv_palette = cv2.merge([h_palette_applied, s, v]) # Image

# Step 7.1 Remap the new palette to the original image
h_palette_remap = remap(h, h_reduced, h_palette_applied)
image_hsv_palette_remap = cv2.merge([h_palette_remap, s, v])

# Step 8: Using the original image, map the palette
image_palette = cv2.cvtColor(image_hsv_palette, cv2.COLOR_HSV2BGR)
image_palette_remap = cv2.cvtColor(image_hsv_palette_remap, cv2.COLOR_HSV2BGR)

print(color_mapping)
cv2.imwrite('reduced_img.jpg', image_reduced)
cv2.imwrite('random_palette_reduced.jpg', image_palette)
cv2.imwrite('random_palette_remap.jpg', image_palette_remap)
