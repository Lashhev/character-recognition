import cv2
import numpy
import matplotlib.pyplot as plt
# from .perceptron import NeuralNetwork
# from .image_codificator import ImageCodificator

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
        self.symbol = None
        # self.neural_network = NeuralNetwork(64)
        self.symbols_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                             'Y', 'Z','0', '1', '2','3','4', '5','6',
                             '7', '8','9']
        cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
        cv2.imshow('Image', self.colored_image)
        cv2.createTrackbar('thresholdR','Binary',255,255,self.__on_change)
        cv2.createTrackbar('thresholdG','Binary',0,255,self.__on_change)
        cv2.createTrackbar('thresholdB','Binary',0,255,self.__on_change)
        self.__on_change(50)

    def recognize(self):
        cv2.imwrite('symbol.png', self.symbol)
        #---Comparison with database---#
        myDatabase = open('base.txt', 'r')
        #-Array of lines-#
        allDatabaseLines = myDatabase.readlines()
        #-Similarity Indicator-#
        similarityIndicator = [0] * len(allDatabaseLines)
        #-Create an array of similarity indicators-#
        for k in range(len(allDatabaseLines)):
            count = int(0)
            for i in range(0, 8):
                for j in range(0, 8):
                    arrayDatabaseLinesHash = allDatabaseLines[k]
                    arrayDatabaseLinesHash = [int(x) for x in str(arrayDatabaseLinesHash[0:64])]
                    if arrayDatabaseLinesHash[count] == 1:
                        arrayDatabaseLinesHash[count] = 255
                    if self.symbol[i, j] == arrayDatabaseLinesHash[count]:
                        similarityIndicator[k] += 1
                    count += 1
        cv2.waitKey()
        #---Searching of character---#
        maxNumber = 0
        for i in range(len(similarityIndicator)):
            if similarityIndicator[i] > similarityIndicator[maxNumber]:
                maxNumber = i
        print(allDatabaseLines[maxNumber])
        myDatabase.close()
        answer = '0'
        answer2 = '0'
        answer3 = '0'
        #---Database updatng---#
        if similarityIndicator[maxNumber] == 64:
            print("The symbol is " + allDatabaseLines[maxNumber][65:66])
        else:
            while answer != 'Y' and answer != 'N' and answer != 'y' and answer != 'n':
                print("The symbol is " + allDatabaseLines[maxNumber][65:66] + "? (Y/N)")
                answer = input()

            if answer == 'Y' or answer == 'y':
                while answer2 != 'Y' and answer2 != 'N' and answer2 != 'y' and answer2 != 'n':
                    print("Do you want to add the character matrix to the database?")
                    answer2 = input()
                if answer2 == 'Y' or answer2 == 'y':
                    myDatabase = open('base.txt', 'a+')
                    myDatabase.write('\n')
                    for i in range(0, 8):
                        for j in range(0, 8):
                            if self.symbol[i, j] == 255:
                                myDatabase.write('1')
                            else:
                                myDatabase.write('0')
                    myDatabase.write('\t')
                    myDatabase.write(allDatabaseLines[maxNumber][65:66])
                    myDatabase.close()
                    print("The symbol was added to database")
                else:
                   pass

            else:
                while answer3 != 'Y' and answer3 != 'N' and answer3 != 'y' and answer3 != 'n':
                    print("Do you want to add the character matrix to the database?")
                    answer3 = input()
                if answer3 == 'Y' or answer3 == 'y':
                    print("What character it is?")
                    charAnswer = input()
                    myDatabase = open('base.txt', 'a+')
                    myDatabase.write('\n')
                    for i in range(0, 8):
                        for j in range(0, 8):
                            if self.symbol[i, j] == 255:
                                myDatabase.write('1')
                            else:
                                myDatabase.write('0')
                    myDatabase.write('\t')
                    myDatabase.write(charAnswer)
                    myDatabase.close()
                    print("The symbol was added to database")
        cv2.destroyAllWindows()

    def run(self):
        while(True):
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        self.recognize()

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
        try:
            BB_img, img_colored_ROI, img_binary_ROI = self.__create_Bounding_Box(binary_img)
            cv2.imshow('Bounding Box image', BB_img)
            cv2.imshow('ROI_colored', img_colored_ROI)
            cv2.imshow('ROI_binary', img_binary_ROI)
            character_sample = self.__get_character_sample(img_binary_ROI)
            cv2.imshow('character_sample', character_sample)
            self.symbol = character_sample
        except:
            print('failed to create bounding')

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
        if up + 3 < image.shape[1] and down - 3 < image.shape[1] and left - 3 > 0 and left - 3 < image.shape[0] and right + 3 < image.shape[0]:
            up+=3
            down -=3
            left-=3
            right+=3
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
        symbol = cv2.resize(binary_image,(8,8), cv2.INTER_LINEAR)
        mask = (symbol < 125)
        imgBinary = numpy.array(255*mask,dtype=numpy.uint8)
        imgBinary = numpy.reshape(imgBinary,(8, 8))
        return imgBinary # TODO: make up an custom algorithm
        



# results = numpy.zeros(len(self.symbols_list))
        # k =0
        # for sym in self.symbols_list:
        #     self.neural_network.load_weights('/home/lashhev/Documents/classificator/weights2.yaml', sym)
        #     result = self.neural_network.think(self.symbol.flatten()//255)
        #     results[k] = result
        #     k+=1
        # ids = numpy.where(results == 1.0)
        # answer = 0
        # for id in ids:
        #     print('Is it \'{:s}\'? [Y/n]'.format(self.symbols_list[id]))
        #     while(answer != 27):
        #         answer = cv2.waitKey(0)
        #         if answer == ord('Y') and answer == ord('y'):
        #             return None
        #         elif answer == ord('N') and answer == ord('n'):
        #             break
        # answer = 0
        # print('Do you want to save this image to dataset?')
        # while(answer != 27):
        #     answer = cv2.waitKey(0)
        #     if answer == ord('Y') and answer == ord('y'):
        #         print('Please input it\'s name')
        #         answer = cv2.waitKey(0)
        #         codificator = ImageCodificator()
        #         with open('/home/lashhev/Documents/classificator/database2/dataset2.csv', 'a') as dataset:
        #             codificator.image_dump(dataset,self.symbol, sym)
        #             return None
        #     elif answer == ord('N') and answer == ord('n'):
        #         break