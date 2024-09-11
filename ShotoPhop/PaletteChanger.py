from random import randint, choice, seed
from itertools import permutations
from collections import Counter
import numpy as np
import cv2

from .ImageHandler import ImageHandler


class PaletteChanger(ImageHandler):
    '''
    Class for handling operations for changing palette on an image
     _----------_
    | Attributes |
     -__________-
    image_file : str
        The file path og the image that will be manipulated
    image : numpy.ndarray
        The image that will be manipulated
    quant_method : str
        The method used for the quantization (default 'step')
    num_colors : int
        The number of colors used for the quantization and palette
    output_file : str
        The name of the output file
     _-------_
    | Methods |
     -_______-
    quantize_hues(hue_channel)
        Reduce the number of distinct values in the hue channel
    save_image(image, name, mode='hsv')
        Converts and image to BGR and saves it
    load_atributes(self, args)
        Set up the class atributes
    apply_palette(hue_channel, palette_mapping)
        Apply a palette to an image
    remap(hue_channel_original, hue_channel_quantized, hue_channel_new)
        Remaps all the colors of the new map to the right shades from the original image
    main()
        Runs the steps for applying a new palette to the image
    '''

    def load_atributes(self, args):
        '''
        Set up the class atributes from a parser object
         ------------
        | Parameters |
         ------------
        args : ArgumentParser
            The argument parser of the main program
        '''
        if not args.seed is None:
            seed(args.seed)
        if not args.palette is None:
            self.num_colors = len(args.palette)
            # Select random permutaion of the palette
            self.palette = choice(list(permutations(args.palette)))
        else:
            self.num_colors = args.palette_size
            self.palette = args.palette
        self.output_file = args.output
       

    def apply_palette(self, hue_channel, palette_mapping):
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
    def remap(self, hue_channel_original, hue_channel_quantized, hue_channel_new):
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
            return `hue_channel_new - (hue_channel_original - hue_channel_quantized)`
        '''
        return hue_channel_new - (hue_channel_original - hue_channel_quantized)


    def main(self):
        '''
        Run the steps for changing the color palette of an image
        Saves three images:
            `quantized_img.jpg` is the image with the color representation reduced
            `quantized_output.jpg` is the image with the new palette with and the
                color representation reduced
            `output.jpg` is the image remapped to the new palette
        '''
        print(f"Input file: {self.image_file}\nPalette size: {self.num_colors}\nQuantization method: `{self.quant_method}`")
        if self.palette is None:
            print("Palette: random")
        else:
            print(f"Palette: {self.palette}")
        # Change the image to HSV
        image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)

        # Reduce the representation of colors
        h_quantized = self.quantize_hues(h)
        
        # Apply the reduced colors to the image
        image_hsv_quantized = cv2.merge([h_quantized, s, v])

        # Get the unique values
        unique_hues = np.unique(h_quantized)

        # Mat the new palette to the original colors
        if self.palette is None: # Randomize palette if there is none
            color_mapping = [(i, randint(0, 180)) for i in unique_hues]
            print('Palette = ' + ','.join(str(i[1]) for i in color_mapping))
        else:
            color_mapping = [(i, new_color) for i, new_color in zip(unique_hues, self.palette)]
        
        # Apply a palette to the image
        h_palette_applied = self.apply_palette(h_quantized.copy(), color_mapping)
        image_hsv_palette = cv2.merge([h_palette_applied, s, v]) # Image

        #  Remap the new palette to the original image
        h_palette_remap = self.remap(h, h_quantized, h_palette_applied)
        image_hsv_palette_remap = cv2.merge([h_palette_remap, s, v])

        # Save results
        extension = self.image_file.split('.')[1]
        if self.output_file is None:
            self.output_file = 'output.' + extension
        self.save_image(image_hsv_quantized, f'quantized_img.{extension}', mode='hsv')
        self.save_image(image_hsv_palette, f'quantized_{self.output_file}', mode='hsv')
        self.save_image(image_hsv_palette_remap, self.output_file, mode='hsv')
