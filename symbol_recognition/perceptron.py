import numpy
class Perceptron:
    # Инициализация нашей  нейронной сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate): #В параметрах мы записываем входные данные,  данные  скрытого слоя, выходные данные ,скорость обучения соответственно)
        # устанавливаем количество узлов входного , скрытого слоя, выходного слоя
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # Тут обозначены веса матрицы, wih -  вес между входным и скрытым слоем , а  так же  who- вес между скрытым и выходным  слоем
        self.wih = numpy.random.rand(self.hnodes, self.inodes)
        self.who = numpy.random.rand(self.onodes, self.hnodes)

        # Скорость обучения -это наш гиперпараметр, то есть, параметр , который мы подбираем ручками, и в зависимости от того, как нам это удобно нам, и , конечно же, нейронной сети
        self.lr = learningrate
        
        # Наша Сигмоида- функция активации
        self.activation_function = lambda x: 1/(1+numpy.exp(-x))

    def train(self, inputs_list, targets_list):
        # Конвертируем наш список в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T # поступающие на вход данные input
        targets = numpy.array(targets_list, ndmin=2).T #целевые значения targets
        
        # Подсчет сигнала в скрытом слое
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Подсчет сигналов, выходящих из скрытого слоя к выходному слою. Тут в нашем узле, куда поступали все данные в переменную hidden_inputs (1я функция), эта переменная подается  как параметр в Сигмоиду - функцию активации (2я функция)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # Подсчет сигналов в конечном(выходном) слое
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Подсчет  сигналов, подающихся в функцию активации
        final_outputs = self.activation_function(final_inputs)
        
        # Значение ошибки (Ожидание - Реальность)
        output_errors = targets - final_outputs
        # Ошибка скрытого слоя становится ошибкой ,которую мы получили для <b>ошибки выходного слоя</b>, но уже <b>распределенные по весам между скрытым и выходным слоями</b>(иначе говоря с учетом умножения соответствующих весов) 
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # Обновление весов между скрытым слоем и выходным (Явление того, что люди зовут ошибкой обратного распространения)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # Обновление весов между скрытым слоем и входным(Та же ошибка ошибка обратного распространения в действии)
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    #Создаем функцию , которая будет принимать входные данные
    def query(self, inputs_list):
        # Конвертируем поданный список входных данных в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Подсчет сигналов в скрытом слое
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Подсчет сигналов, поданных в функцию активации
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #Подсчет сигналов в конечном выходном слое
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #Подсчет сигналов в конечном выходном слое, переданных в функцию активации
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs