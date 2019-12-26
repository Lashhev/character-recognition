#!/usr/bin/python3
import cv2
import csv
from glob import glob
import os
import numpy as np
from symbol_recognition import to_binary_img

class ImageCodificator:
    def __init__(self):
        self.symbols_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                             'Y', 'Z','0', '1', '2','3','4', '5','6',
                             '7', '8','9']
    
    def load_image(self, filename:str):
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    def threshold(self, image, threshold:int):
        return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    def threshold_inv(self, image, threshold:int):
        return cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

    def get_image_names(self, directory:str):
        return glob(os.path.join(directory,'*.png'))

    def prepair_train_sample(self, symbol_key:str, symbol_image):
        binary_data = [ord(symbol_key)]
        symbol_image = cv2.resize(symbol_image,(8,8))
        _, symbol_image = self.threshold(symbol_image, 125)
        cv2.imshow('binary', symbol_image)
        cv2.waitKey(0)
        binary_data.extend(symbol_image.ravel()//255)
        return binary_data

    def image_dump(self, stream, image, symbol):
        csvwriter = csv.writer(stream, delimiter=',',
                            quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        binary_data = self.prepair_train_sample(symbol, image)

        csvwriter.writerow(binary_data)

    def image_save(self, filename, image):
        cv2.imwrite(filename, image)
   
    def encode(self, in_directory:str, out_directory:str):
        with open(os.path.join(out_directory, 'train_set.csv'), 'a') as csv_file:
            for symbol in self.symbols_list:
                path = os.path.join(in_directory, symbol)
                imade_names = self.get_image_names(path)
                for image_name in imade_names:
                    image = self.load_image(image_name)
                    _ , image_binary = self.threshold(image, 125)
                    
                    self.image_dump(csv_file, image_binary, symbol)
                    


if __name__ == "__main__":
    encoder = ImageCodificator()
    encoder.encode('/home/lashhev/Documents/classificator/database', 
                            '/home/lashhev/Documents/classificator/database')