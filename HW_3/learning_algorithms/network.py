"""
Модифицированный network.py с сайта и книги neuralnetworksanddeeplearning.com
Модуль, реализующий алгоритм стохастического градиентного спуска
для обыкновенной нейронной сети. Градиенты вычисляются с помощью
алгоритма обратного распространения ошибки. Хочу отметить, что я,
в первую очередь, старался сделать код простым, легко читаемым и
легко изменяемым. Он не оптимизирован и в нём не реализованы многие
из желаемых возможностей.
"""

import random
import numpy as np


#### Вспомогательные функции
def sigmoid(z):
    """
    Сигмоида
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Производная сигмоиды по e (шутка). По x
    """
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes, output_function=sigmoid, output_derivative=sigmoid_prime):
        """
        Список ``sizes`` содержит количество нейронов в соответствующих слоях
        нейронной сети. К примеру, если бы этот лист выглядел как [2, 3, 1],
        то мы бы получили трёхслойную нейросеть, с двумя нейронами в первом
        (входном), тремя нейронами во втором (промежуточном) и одним нейроном
        в третьем (выходном, внешнем) слое. Смещения и веса для нейронных сетей
        инициализируются случайными значениями, подчиняющимися стандартному нормальному
        распределению. Обратите внимание, что первый слой подразумевается слоем,
        принимающим входные данные, поэтому мы не будем добавлять к нему смещение
        (делать это не принято, поскольку смещения используются только при
        вычислении выходных значений нейронов последующих слоёв).
        Параметры output_function и output_derivative задают активационную функцию
        нейрона выходного слоя и её производную.
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        assert output_function is not None, "You should either provide output function or leave it default!"
        self.output_function = output_function
        assert output_derivative is not None, "You should either provide derivative of the output function or leave it default!"
        self.output_derivative = output_derivative

    def feedforward(self, a):
        """
        Вычислить и вернуть выходную активацию нейронной сети
        при получении ``a`` на входе (бывшее forward_pass).
        """
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)

        output = np.dot(self.weights[-1], a) + self.biases[-1]
        output = self.output_function(output)

        return output

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda_l1=0.0, lmbda_l2=0.0):
        """
        Обучить нейронную сеть, используя алгоритм стохастического
        (mini-batch) градиентного спуска.
        ``training_data`` - лист кортежей вида ``(x, y)``, где
        x - вход обучающего примера, y - желаемый выход (в формате one-hot).
        Роль остальных обязательных параметров должна быть понятна из их названия.
        Если предоставлен опциональный аргумент ``test_data``,
        то после каждой эпохи обучения сеть будет протестирована на этих данных
        и промежуточный результат обучения будет выведен в консоль.
        ``test_data`` -- это список кортежей из входных данных
        и номеров правильных классов примеров (т.е. argmax(y),
        если y -- набор ответов в той же форме, что и в тренировочных данных).
        Тестирование полезно для мониторинга процесса обучения,
        но может существенно замедлить работу программы.
        """
        loss_history = []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda_l1, lmbda_l2)
            print("Epoch {0} complete".format(j))
            loss = self.total_loss(training_data, lmbda_l1, lmbda_l2)
            loss_history.append(loss)

        return loss_history

    def update_mini_batch(self, mini_batch, eta, lmbda_l1=0.0, lmbda_l2=0.0):
        """
        Обновить веса и смещения нейронной сети, сделав шаг градиентного
        спуска на основе алгоритма обратного распространения ошибки, примененного
        к одному mini batch.
        ``mini_batch`` - список кортежей вида ``(x, y)``,
        ``eta`` - величина шага (learning rate),
        ``lmbda_l1`` - коэффициент регуляризации L1,
        ``lmbda_l2`` - коэффициент регуляризации L2.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta / len(mini_batch)
        
        # L1 регуляризация
        self.weights = [(1 - eps * lmbda_l1) * w - eps * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        
        # L2 регуляризация
        self.weights = [(1 - eps * lmbda_l2) * w - eps * nw 
                        for w, nw in zip(self.weights, nabla_w)]
        
        self.biases = [b - eps * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Возвращает кортеж ``(nabla_b, nabla_w)`` -- градиент целевой функции по всем параметрам сети.
        ``nabla_b`` и ``nabla_w`` -- послойные списки массивов ndarray,
        такие же, как self.biases и self.weights соответственно.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # прямое распространение
        activation = x
        activations = [x]  # лист, хранящий все активации, слой за слоем
        zs = []  # лист, хранящий все z векторы, слой за слоем

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        output = self.output_function(z)
        activations.append(output)

        # обратное распространение
        delta = self.cost_derivative(activations[-1], y) * self.output_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Обратите внимание, что переменная l в цикле ниже используется
        # немного иначе, чем в лекциях.  Здесь l = 1 означает последний слой,
        # l = 2 - предпоследний и так далее.
        # Мы перенумеровали схему, чтобы с удобства для себя
        # использовать тот факт, что в Python к переменной типа list
        # можно обращаться по негативному индексу.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Вернуть количество тестовых примеров, для которых нейронная сеть
        возвращает правильный ответ. Обратите внимание: подразумевается,
        что выход нейронной сети - это индекс, указывающий, какой из нейронов
        последнего слоя имеет наибольшую активацию.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Возвращает вектор частных производных (\partial C_x) / (\partial a)
        целевой функции по активациям выходного слоя.
        """
        return output_activations - y

    def total_loss(self, training_data, lmbda_l1=0.0, lmbda_l2=0.0):
        """
        Вычислить общую ошибку на обучающих данных с учетом регуляризации.
        ``training_data`` - список кортежей вида ``(x, y)``,
        ``lmbda_l1`` - коэффициент регуляризации L1,
        ``lmbda_l2`` - коэффициент регуляризации L2.
        """
        loss = 0.0
        for x, y in training_data:
            output = self.feedforward(x)
            loss += 0.5 * np.sum((output - y) ** 2)
        
        # L1 регуляризация
        loss += lmbda_l1 * sum(np.sum(np.abs(w)) for w in self.weights)
        
        # L2 регуляризация
        loss += 0.5 * lmbda_l2 * sum(np.sum(np.square(w)) for w in self.weights)
        
        return loss / len(training_data)
