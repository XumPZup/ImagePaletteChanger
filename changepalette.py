from argparse import ArgumentParser
from ShotoPhop.PaletteChanger import PaletteChanger


# Util for getting the command parameters
def get_parameters():
    '''
    Utility function to get termianl arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-i', '--image', type=str, help='Input image file', required=True)
    parser.add_argument('-s', '--palette-size', type=int, nargs='+', help='Size of the palette (two values for method hybrid', required=True)
    parser.add_argument('-p', '--palette', type=int, nargs='+', help='Palette')
    parser.add_argument('-m', '--method', type=str, help='Quantization method (default setp)')
    parser.add_argument('-o', '--output', type=str, help='Output file name')
    parser.add_argument('--seed', type=int, help='Seed for the randomization')
    parser.add_argument('--debug', type=int, help='Set to 1 for saving debug images')

    return parser.parse_args()

args = get_parameters()
handler = PaletteChanger(args.image, args.palette_size, args.method, args.debug)
handler.load_atributes(args)
handler.main()
