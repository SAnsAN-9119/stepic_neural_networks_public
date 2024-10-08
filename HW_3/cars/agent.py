# cars\agent.py
import os
import random
import sys

# Добавляем путь к родительской директории
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from cars.utils import Action
from learning_algorithms.network import Network


class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):
    def __init__(self, history_data=int(50000)):
        """
        Создаёт машинку
        :param history_data: количество хранимых нами данных о результатах предыдущих шагов
        """
        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self._rays = 9  # выберите число лучей ладара; например, 5
        # Использовать перемещённые функции вместо lambda
        self.neural_net = Network(
            [self.rays+4,
             16, 32, 16,
             1],
            output_function=output_function,
            output_derivative=output_derivative
        )
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0
        self.loss_history = []
        self.update_plots = False

    @classmethod
    def from_weights(cls, layers, weights, biases):
        """
        Создание агента по параметрам его нейронной сети. Разбираться не обязательно.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        nn = Network(layers, output_function=lambda x: x, output_derivative=lambda x: 1)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # это важный импорт, без которого не пройдёт нормально eval
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        c = open(filename, "r").read()
        return cls.from_string(c)

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        np.set_printoptions(threshold=sys.maxsize)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        """
        Выбирает лучшее действие на основе предсказанных наград от нейронной сети.

        Параметры:
            sensor_info: Текущее состояние окружающей среды.

        Возвращает:
            Выбранное действие, которое, как ожидается, приведет к самой высокой награде.
        """
        # хотим предсказать награду за все действия, доступные из текущего состояния
        rewards_to_controls_map = {}
        # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
        for steering in np.linspace(-1, 1, 3):  # выбирать можно и другую частоту дискретизации, но
            for acceleration in np.linspace(-0.75, 0.75, 3):  # в наших тестах будет именно такая
                action = Action(steering, acceleration)
                agent_vector_representation = np.append(sensor_info, action)
                agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                rewards_to_controls_map[predicted_reward] = action

        # ищем действие, которое обещает максимальную награду
        rewards = list(rewards_to_controls_map.keys())
        highest_reward = max(rewards)
        best_action = rewards_to_controls_map[highest_reward]

        # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
        # рандомное действие
        if (not self.evaluate_mode) and (random.random() < 0.05):
            highest_reward = rewards[np.random.choice(len(rewards))]
            best_action = rewards_to_controls_map[highest_reward]
            # следующие строки помогут вам понять, что предсказывает наша сеть
            print("Chosen random action w/reward: {}".format(highest_reward))
        else:
            print("Chosen action w/reward: {}".format(highest_reward))

        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  # мы пока не знаем, какая будет награда, это
        # откроется при вызове метода receive_feedback внешним миром

        return best_action

    def receive_feedback(self, reward, train_every=20, reward_depth=25):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        :param train_every: сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        :param reward_depth: на какую глубину по времени распространяется полученная награда
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1

        # Добавляем награду, если машина не врезалась в стены последние reward_depth//2 шагов
        if all(r >= 0 for r in list(self.reward_history)[-reward_depth // 2:]):
            self.reward_history[-1] += 1  # Добавляем положительную награду

        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= train_every выполнялось
        if not self.evaluate_mode and (len(self.reward_history) >= train_every) and not (self.step % train_every):
            X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            y_train = np.array(self.reward_history)

            # Преобразуем данные в нужный формат
            train_data = [(x.reshape(-1, 1), np.array([y])) for x, y in zip(X_train, y_train)]
            loss_history = self.neural_net.SGD(training_data=train_data, epochs=10, mini_batch_size=32, eta=0.005,
                                               lmbda_l1=0.0005, lmbda_l2=0.005)
            self.loss_history.extend(loss_history)
            self.update_plots = True

    def plot_loss(self, ax_full, ax_recent):
        ax_full.clear()
        ax_full.plot(self.loss_history)
        ax_full.set_xlabel('Iteration')
        ax_full.set_ylabel('Loss')
        ax_full.set_title('Training Loss (Full History)')

        recent_loss = self.loss_history[-30:]
        iterations = range(len(self.loss_history) - len(recent_loss), len(self.loss_history))
        ax_recent.clear()
        ax_recent.plot(iterations, recent_loss)
        ax_recent.set_xlabel('Iteration')
        ax_recent.set_ylabel('Loss')
        ax_recent.set_title('Training Loss (Last 2 Epochs)')
        
        # Корректируем размер шкалы в соответствии с текущими значениями
        ax_recent.set_ylim(min(recent_loss) - 0.1, max(recent_loss) + 0.1)

    def get_training_params(self):
        return self.neural_net.get_training_params()

def output_function(z):
    return z


def output_derivative(z):
    return 1
