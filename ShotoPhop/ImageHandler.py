from collections import Counter
import cv2


class ImageHandler:
    '''
    Base class for handling operations on image pixels
     _----------_
    | Attributes |
     -__________-
    image_file : str
        The file path of the image that will be manipulated
    image : numpy.ndarray
        The image that will be manipulated
    num_colors : int
        The number of colors used for the quantization
    quant_method : str
        The method used for the quantization (default 'step')
     _-------_
    | Methods |
     -_______-
    quantize_hues(hue_channel)
        Reduce the number of distinct values in the hue channel
    save_image(image, name, mode='hsv')
        Converts and image to BGR and saves it
    '''

    def __init__(self, image_file, num_colors, quant_method='step'):
        '''
         ------------
        | Parameters |
         ------------
        image : str
            The file path of the image that will be manipulated
        num_colors : int
            The number of colors used for the quantization
        quant_method : str
            The method used for the quantization (default 'step')
            Can use the methods:
                'step' : divides the color space in n equal sectors
                'freq' : quantize the hue based on the n most popular values
        '''
        self.image_file = image_file
        self.image = cv2.imread(self.image_file)
        self.quant_method = quant_method

    # Function to map hue to a smaller palette
    def quantize_hues(self, hue_channel):
        '''
        Quantize the hue_channel of an image reducing it to a defined number of colors
        Can use the methods:
            'step' : divides the color space in n equal sectors
            'freq' : quantize the hue based on the n most popular values
        The method is pecified during the initialization of the class and
        it's default value is 'step'
         ------------
        | Parameters |
         ------------
        hue_channel : numpy.ndarray
            A 2D array correspondig to the hue_channel of an image
         ---------
        | Returns |
         ---------
         Out : numpy.darray
            The quantized hue channel
        '''
        # Quantize the hue values to the nearest palette value
        if self.quant_method == 'step':
            max_hue = 180
            interval = max_hue // self.num_colors
            quantized_hues = (hue_channel // interval) * interval
        # Quantize based on color popularity (get the n most frequent colors)
        elif self.quant_method == 'freq':
            # Get the most frequent colors ordered by their value
            most_frequents = sorted(
                    Counter(hue_channel.flatten()).most_common(self.num_colors),
                    key=lambda x: x[0]
                    )
            # Map colors down to the nearest most frequent value
            i = 0
            quantized_hues = hue_channel.copy()
            while i < self.num_colors - 1:
                quantized_hues[(quantized_hues >= most_frequents[i][0]) & (quantized_hues < most_frequents[i+1][0])] = most_frequents[i][0]
                i+=1
            # If the smallest value is not 0 then map all the values below to the highest
            if most_frequents[0][0] != 0:
                quantized_hues[quantized_hues < most_frequents[0][0]] = most_frequents[i][0]
            quantized_hues[quantized_hues >= most_frequents[i][0]] = most_frequents[i][0]

        return quantized_hues

        
    def save_image(self, image, name, mode):  
        '''
        Convert and save an image
        ------------
        | Parameters |
         ------------
        image : np.ndarray
            The image that will be saved
        name : str
            The name of the image file
        mode : str
            The mode of the image (hsv, hls, bgr)
        '''
        if mode == 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif mode == 'hls':
            image = cv2.cvtColor(image, cv2.COLOR_HLS2BGR)

        cv2.imwrite(name, image)
