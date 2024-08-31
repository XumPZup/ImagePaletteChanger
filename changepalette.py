from itertools import permutations
from collections import Counter
from random import randint, choice
import numpy as np
from argparse import ArgumentParser
import cv2

# Util for getting the command parameters
def get_parameters():
    '''
    Utility function to get termianl arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Input image file', required=True)
    parser.add_argument('-s', '--palette-size', type=int, help='Size of the palette')
    parser.add_argument('-p', '--palette', type=int, nargs='+', help='Palette')
    parser.add_argument('-m', '--method', type=str, help='Quantization method (default setp)')
    parser.add_argument('-o', '--output', type=str, help='Output file name')

    args = parser.parse_args()
    quant_method = args.method
    if quant_method is None:
        quant_method = 'step'
    if not args.palette is None:
        # Return arguments with random permutation of the palette
        return args.image, len(args.palette), choice(list(permutations(args.palette))), args.output, quant_method
    return args.image, args.palette_size, args.palette, args.output, quant_method
        

# Function to map hue to a smaller palette
def quantize_hues(hue_channel, num_colors, method='step'):
    '''
    Quantize the hue_channel of an image reducing it to a defined number of colors
     ------------
    | Parameters |
     ------------
    hue_channel : np.ndarray
        A 2D array correspondig to the hue_channel of an image
    num_colors : int
        The number of distinct colors for the restul image
    method : str (default 'step')
        Metod used for the quantization
        'step' : divides the color space in n equal sectors
        'freq' : quantize the hue based on the n most popular values
     --------- 
    | Returns |
     ---------
     Out : ndarray
        The quantized hue channel
    '''
    # Quantize the hue values to the nearest palette value
    if method == 'step':
        max_hue = 180
        interval = max_hue // num_colors
        quantized_hues = (hue_channel // interval) * interval
    # Quantize based on color popularity (get the n most frequent colors)
    elif method == 'freq':
        # Get the most frequent colors ordered by their value
        most_frequents = sorted(
                Counter(hue_channel.flatten()).most_common(num_colors), 
                key=lambda x: x[0]
                )
        # Map colors down to the nearest most frequent value 
        i = 0
        quantized_hues = hue_channel.copy()
        while i < num_colors - 1:
            quantized_hues[(quantized_hues >= most_frequents[i][0]) & (quantized_hues < most_frequents[i+1][0])] = most_frequents[i][0]
            i+=1
        # If the smallest value is not 0 then map all the values below to the highest
        if most_frequents[0][0] != 0:
            quantized_hues[quantized_hues < most_frequents[0][0]] = most_frequents[i][0]
        quantized_hues[quantized_hues >= most_frequents[i][0]] = most_frequents[i][0]
        
    return quantized_hues


def apply_palette(hue_channel, palette_mapping):
    '''
    Change colors of an image hue_channel mapping the original palette to a new one
     ------------
    | Parameters |
     ------------
    hue_channel : np.ndarray
        A 2D array correspondig to the hue_channel of an image
    palette_mapping : list
        A list of tuples each tuple contains the pair (original, new) where
        `original` is one of the values of the palette of the input image and
        `new` is one of the values of the new palette
     --------- 
    | Returns |
     ---------
     Out : ndarray
        The hue channel remapped to the new palette
    '''
    # Map the hues to the new palette
    for original, new in palette_mapping:
        hue_channel[hue_channel == original] = new
    return hue_channel

# Remaps the colors using distance from the original
def remap(hue_channel_original, hue_channel_quantized, hue_channel_new):
    '''
    Expands the color space of the a remapped image based on the distance of 
    the original and quantized hue channel values
     ------------
    | Parameters |
     ------------
    hue_channel_original : np.ndarray
        The hue channel of the input image
    hue_channel_quantized : np.ndarray
        The hue channel of the input image quantized
    hue_channel_new : np.ndarray
        The hue channel remapped to a new palette
     --------- 
    | Returns |
     ---------
     Out : ndarray
        The hue channel remapped to the origianl image hues
    '''
    return hue_channel_new - (hue_channel_original - hue_channel_quantized)

if __name__ == '__main__':
    # Step 0: Get parameters
    file_name, num_colors, user_palette, output_file, quant_method = get_parameters()
    print(f"Input file: {file_name}\nPalette size: {num_colors}\nQuantization method: `{quant_method}`")
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
    h_quantized = quantize_hues(h, num_colors, method=quant_method)

    # Step 5: Apply the new colors to the image
    image_hsv_quantized = cv2.merge([h_quantized, s, v])
    image_quantized = cv2.cvtColor(image_hsv_quantized, cv2.COLOR_HSV2BGR)

    # Step 6: Find the n most popular colors
    # n_colors = 5
    h_flat = h_quantized.flatten()
    color_counts = Counter(h_flat)
    most_common_hues = color_counts.most_common(num_colors) # Not neaded since I'm using the same colors as in `h_quantized`

    # Step 7: Apply a palette to the image
    if user_palette is None: # Randomize palette if there is none
        color_mapping = [(i[0], randint(0, 180)) for i in most_common_hues]
    else:
        color_mapping = [(i[0], new_color) for i, new_color in zip(most_common_hues, user_palette)]

    h_palette_applied = apply_palette(h_quantized.copy(), color_mapping)
    image_hsv_palette = cv2.merge([h_palette_applied, s, v]) # Image

    # Step 7.1 Remap the new palette to the original image
    h_palette_remap = remap(h, h_quantized, h_palette_applied)
    image_hsv_palette_remap = cv2.merge([h_palette_remap, s, v])

    # Step 8: Using the original image, map the palette
    image_palette = cv2.cvtColor(image_hsv_palette, cv2.COLOR_HSV2BGR)
    image_palette_remap = cv2.cvtColor(image_hsv_palette_remap, cv2.COLOR_HSV2BGR)

    # Save results
    if output_file is None:
        extension = file_name.split('.')[1]
        output_file = 'output.' + extension
    cv2.imwrite('quantized_img.jpg', image_quantized)
    cv2.imwrite(f'quantized_{output_file}', image_palette)
    cv2.imwrite(output_file, image_palette_remap)
