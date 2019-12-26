import sys
import argparse
from symbol_recognition import SymbolClassificator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Сharacter recognition.')
    parser.add_argument('image', type=str, default='testR3.png',help='Image file name')
    args = parser.parse_args()
    symbolClassificator = SymbolClassificator(args.image)
    symbolClassificator.run()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
