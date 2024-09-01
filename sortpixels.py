from argparse import ArgumentParser

from ShotoPhop.PixelSorter import PixelSorter


# Util for getting the command parameters
def get_parameters():
    '''
    Utility function to get termianl arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Input image file', required=True)
    parser.add_argument('-s', '--size', type=int, help='Number of different colors used during the extraction of the region', required=True)
    parser.add_argument('-r', '--range', type=int, nargs='+', help='The hue range for the mask', required=True)
    parser.add_argument('-m', '--method', type=str, help='Quantization method (default setp)')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('--mode', type=str, help='(Default `rgb`) The mode used for the sorting (rgb, hsv, hls)')

    args = parser.parse_args()
     
    return args


args = get_parameters()
handler = PixelSorter(args.image, args.size, args.method)
handler.load_atributes(args)
handler.main()
