import numpy as np
import cv2

from .ImageHandler import ImageHandler


class PixelSorter(ImageHandler):
    '''
    Class for handling operations for changing palette on an image
     _----------_
    | Attributes |
     -__________-
    image_file : str
        The file path og the image that will be manipulated
    image : numpy.ndarray
        The image that will be manipulated
    range : list
        A pair of values used to filter the pixel to sort
    quant_method : str
        The method used for the quantization (default 'step')
    num_colors : int
        The number of colors used for the quantization
    mode : str
        The image mode used for the pixel sorting (hsv, hls, bgr)
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
    sort_pixels(hue_channel_quantized,  mode='hsv')
        Sorts the pixels that fall into a certain hue value range
    find_region(image_hsv, hue_channel_quantized, color=[0,0,0]):
        Set the same color to all the pixes that fall in a certain range of hue
    main()
        Runs the steps for sorting the pixels of the image
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
        self.num_colors = args.size
        self.range = args.range
        self.quant_method = 'step' if args.method is None else args.method
        self.mode = 'bgr' if args.mode is None else args.mode
        self.bebug = False if args.debug is None else args.debug
        self.output_file = args.output
       
    
    def sort_pixels(self, hue_channel_quantized):
        '''
        Sort the pixels that fall into a certain hue value range
         ------------
        | Parameters |
         ------------
        hue_channel_quantized : np.ndarray
            A 2D array corresponding to the hue channel of the original image quantized
         ---------
        | Returns |
         ---------
         Out : numpy.ndarray
            The image converted to the specified mode and with its pixels sorted
        '''
        result = self.image.copy()
        # Get indexes
        idx = np.where((hue_channel_quantized >= self.range[0]) & (hue_channel_quantized < self.range[1]))
        # Convert the image to the desired mode
        if self.mode == 'hsv':
            result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        elif self.mode == 'hls':
            result = cv2.cvtColor(result, cv2.COLOR_BGR2HLS)
        # Sort pixels
        filtered_colors = result[idx]
        result[idx] = np.sort(filtered_colors, axis=0)

        return result


    def find_region(self, image_hsv, hue_channel_quantized, color=[0,0,0]):
        '''
        Set the same color to all the pixes that fall in a certain range of hue
         ------------
        | Parameters |
         ------------
        image_hsv : np.ndarray
            The original hsv image
        hue_channel_quantized : np.ndarray
            A 2D array correspondig to the hue_channel of the original image quantized
        color : list
            The HSV color that will set to the pixels that fall in the range
         ---------
        | Returns |
         ---------
         Out : ndarray
            The HSV image with the specified region of the same color
        '''
        result = image_hsv.copy()
        idx = np.where((hue_channel_quantized >= self.range[0]) & (hue_channel_quantized < self.range[1]))
        result[idx] = color

        return result


    def main(self):
        '''
        Run the steps for sorting the pixels of the image
        Saves three images:
            `region.jpg` is the original image with the selected pixels 
                set to the same color
            `quant.jpg` is the image with a reduced color representation
            `output.jpg` is the image with the pixel sorted
        '''
        print(f"Input file: {self.image_file}\nQuantization colors: {self.num_colors}\nQuantization method: `{self.quant_method}`\nMode: `{self.mode}`")
        # Change the image to HSV
        image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        # Quantize the hue channel
        h_quantized = self.quantize_hues(h)
        # Sort the image on a specific region
        sorted_image = self.sort_pixels(h_quantized)
        # Darken the image region
        image_region = self.find_region(image_hsv, h_quantized)

        image_hsv[:,:,0] = h_quantized

        # Save results
        extension = self.image_file.split('.')[1]
        if self.output_file is None:
            self.output_file = 'output.' + extension
        if self.debug:
        # Save quantized image original and with the new palette
            self.save_image(image=image_hsv, name='quant.jpg', mode='hsv')
            self.save_image(image=image_region, name='region.jpg', mode='hsv')
        # Save final result
        self.save_image(image=sorted_image, name=self.output_file, mode=self.mode)
