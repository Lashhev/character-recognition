import cv2
import numpy as np
from glob import glob
import os
import csv
import sys
from image_codificator import ImageCodificator

def is_contour_too_small(P):
    return P > 64 # минимальная площадь образца
def mkdir_if_need(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

class TextParser:
    def __init__(self):
        self.symbol_list = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    def findContours(self, image_path:str = "/home/lashhev/Downloads/symbols1.png"):
        self.image_path = image_path
        self.image_orig = cv2.imread(self.image_path)
        gray = cv2.cvtColor(self.image_orig, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(gray, (5,5), 0) 
        thresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return thresh, contours, hierarchy

    def get_symbols(self, binary_image, contours, hierarchy):
        k = 0
        # symbol_list_rev = self.symbol_list[::-1]
        symbols = list()
        symbol_keys = list()
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            # if hierarchy[0][idx][3] == 255 and (is_contour_too_small(h*w)):
            if (is_contour_too_small(h*w)):
                symbol = binary_image[y:y + h, x: x + w]
                cv2.rectangle(self.image_orig, (x, y), (x + w, y + h), (70, 0, 0), 1)
                
                cv2.imshow('Symbol', symbol)
                # print(symbol.tolist())
                cv2.imshow('Marked', self.image_orig)
                print('What symbol is it?')
                intChar = cv2.waitKey(0)
                if intChar == 27:                   # if esc key was pressed
                    sys.exit(0)                      # exit program
                elif intChar in self.symbol_list:      # else if the char is in the list of chars we are looking for . . .
                    symbol_keys.append(chr(intChar))
                    symbols.append(symbol)
                    print(' \'{:s}\'. Got it!'.format(chr(intChar)))   
                k+=1
        return symbol_keys, symbols

    def count_records(self, dataset_path:str):
        file_names = glob(os.path.join(dataset_path,'*.png'))
        file_names.extend(glob(os.path.join(dataset_path,'*.jpg')))
        return len(file_names)
    
    def save_symbols(self, dataset_path:str, symbol_keys:list, symbol_images:list):
        mkdir_if_need(dataset_path)
        print("How do you want to save found symbols?")
        print('[0] - As image')
        print('[1] - As csv data')
        print('[Esc] - Exit')
        answer = 0
        while(answer != 27):
            answer = cv2.waitKey(0)
            if answer == ord('0'):
                for i in range(0, len(symbol_images)):
                    dir = os.path.join(dataset_path, symbol_keys[i])
                    mkdir_if_need(dir)
                    k = self.count_records(dir)
                    image_path =  os.path.join(dir, '{:d}.png'.format(k + 1))
                    # print(symbol_images[i].tolist())
                    # cv2.waitKey(0)
                    cv2.imwrite(image_path, symbol_images[i])
                break
            elif answer == ord('1'):
                encoder = ImageCodificator()
                with open(os.path.join(dataset_path,'dataset2.csv'), 'a') as dataset:
                    for i in range(0, len(symbol_images)):
                        encoder.image_dump(dataset, symbol_images[i],symbol_keys[i] )
                break

if __name__ == "__main__":
    image_file = "/home/lashhev/Downloads/small.png"
    output_dir = '/home/lashhev/Documents/classificator/database/'
    parser = TextParser()
    binary_image, contours, hierarchy = parser.findContours(image_file)
    symbol_keys, symbols = parser.get_symbols(binary_image, contours, hierarchy)
    parser.save_symbols(output_dir, symbol_keys, symbols)

   