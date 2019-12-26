import cv2
import numpy
import matplotlib.pyplot as plt
emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, \
                66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, \
                78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, \
                97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, \
                108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, \
                119, 120, 121, 122]
def split_to_channels(colored_image):
    #--Red Channel--#
    imgRed = colored_image[:, :, 2]
    imgGreen= colored_image[:, :, 1]
    imgBlue= colored_image[:, :, 0]
    return imgRed, imgGreen, imgBlue

# #--To binary--#
def to_binary_img(colored_image, thresholdBGR):
    rows = colored_image.shape[0]
    cols = colored_image.shape[1]
    # imgBinary = numpy.ndarray([rows, cols], dtype=numpy.uint8)
    imgRed, imgGreen, imgBlue = split_to_channels(colored_image)
    mask = (imgRed < thresholdBGR[2]) & (imgGreen >= thresholdBGR[1]) & (imgBlue >= thresholdBGR[0])
    imgBinary = numpy.array(255*mask,dtype=numpy.uint8)
    imgBinary = numpy.reshape(imgBinary,(rows, cols))
    return imgBinary

class SymbolClassificator:
    def __init__(self, image:str):
        self.colored_image = cv2.imread(image)
        cv2.namedWindow('Binary')
        cv2.imshow('Image', self.colored_image)
        cv2.createTrackbar('thresholdR','Binary',255,255,self.__on_change)
        cv2.createTrackbar('thresholdG','Binary',0,255,self.__on_change)
        cv2.createTrackbar('thresholdB','Binary',0,255,self.__on_change)
        self.__on_change(50)

    def run(self):
        while(True):
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    def __on_change(self, val):
        r = cv2.getTrackbarPos('thresholdR','Binary')
        g = cv2.getTrackbarPos('thresholdG','Binary')
        b = cv2.getTrackbarPos('thresholdB','Binary')
        # ret, binary_img = cv2.threshold(self.colored_image,r, 255, cv2.THRESH_BINARY)
        binary_img = to_binary_img(self.colored_image,[b, g, r])
        # cv2.imshow('Binary', binary_img)
        binary_img = self.__dilate_and_erode(binary_img)
        cv2.imshow('Binary2', binary_img)
        cv2.namedWindow('Binary2',cv2.WINDOW_NORMAL)
        BB_img, img_colored_ROI, img_binary_ROI = self.__create_Bounding_Box(binary_img)
        cv2.imshow('Bounding Box image', BB_img)
        cv2.imshow('ROI_colored', img_colored_ROI)
        cv2.imshow('ROI_binary', img_binary_ROI)
        character_sample = self.__get_character_sample(img_binary_ROI)
        cv2.imshow('character_sample', character_sample)

    def __dilate_and_erode(self, image):
        element1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
        image = cv2.dilate(image, element1)
        image = cv2.erode(image, element2)
        return image

    def __create_Bounding_Box(self, image):
        index_row, index_col = numpy.where(image == 0)
        left = index_col.min()
        right = index_col.max()
        up = index_row.min()
        down = index_row.max()
        imgBB = numpy.copy(self.colored_image)
        cv2.rectangle(imgBB,(left,up),(right,down),(0,255,255),2)
        img_colored_ROI = self.__get_colored_ROI([up, down, left, right])
        img_binary_ROI = self.__get_binary_ROI(image, [up, down, left, right])
        return imgBB, img_colored_ROI, img_binary_ROI

    def __get_colored_ROI(self, rect):
        imgROI = self.colored_image[rect[0]:rect[1], rect[2]:rect[3]]
        return imgROI

    def __get_binary_ROI(self, image, rect):
        imgROI = image[rect[0]:rect[1], rect[2]:rect[3]]
        return imgROI

    def __get_character_sample(self, binary_image):
        return cv2.resize(binary_image,(32,32), cv2.INTER_LINEAR) # TODO: make up an custom algorithm
        



