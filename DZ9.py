import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10 # общее количество классов, в нашем случае это цифры от 0 до 9
num_features = 784 # количество атрибутов входного вектора 28 * 28 = 784

learning_rate = 0.001 # скорость обучения нейронной сети
training_steps = 3000 # максимальное число эпох
batch_size = 256 # пересчитывать веса сети мы будем не на всей выборке, а на ее случайном подможестве из batch_size элементов
display_step = 100 # каждые 100 итераций мы будем показывать текущее значение функции потерь и точности

n_hidden_1 = 128 # количество нейронов 1-го слоя
n_hidden_2 = 256 # количество нейронов 2-го слоя

from tensorflow.keras.datasets import mnist

# Загружаем датасет
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Преобразуем целочисленные пиксели к типа float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# Преобразуем матрицы размером 28x28 пикселей в вектор из 784 элементов
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Нормализуем значения пикселей
x_train, x_test = x_train / 255., x_test / 255.

# Перемешаем тренировочные данные
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

class DenseLayer(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name="w"
        )
        self.b = tf.Variable(tf.zeros([out_features]), name="b")
        print(self.w.shape, self.b.shape)

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.sigmoid(y)


class NN(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    # Первый слой, состоящий из 128 нейронов
    self.layer_1 = DenseLayer(in_features=num_features, out_features=n_hidden_1)
    # Второй слой, состоящий из 256 нейронов
    self.layer_2 = DenseLayer(in_features=n_hidden_1, out_features=n_hidden_2)
    # Выходной слой
    self.layer_3 = DenseLayer(in_features = n_hidden_2, out_features=num_classes)


  def __call__(self, x):
    # Помните что для выхода нейронной сети мы применяем к выходу функцию softmax. 
    # Делаем мы это для того, чтобы
    # выход нейронной сети принимал значения от 0 до 1 в соответствии с вероятностью 
    # принадлежности входного объекта к одному из 10 классов
    y = self.layer_1(x)
    y = self.layer_2(y)
    y = self.layer_3(y)
    return tf.nn.softmax(y)
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)

    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    #print(tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred))))
    # Вычисление кросс-энтропии
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# В качестве метрики качества используем точность
def accuracy(y_pred, y_true):

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    #print(m.result().numpy())
    #return m.result().numpy()
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()

neural_net = NN(name="mnist")
#optimizer = tf.optimizers.SGD(learning_rate)
# Функция обучения нейросети

dw1 = ""  
dw2 = ""  
dw3 = "" 
db1 = "" 
db2 = ""
db3 = "" 

def train(input_x, output_y):
  # Для подгонки весов сети будем использовать стохастический градиентный спуск
  optimizer = tf.keras.optimizers.experimental.SGD(learning_rate)
  

  # Активация автоматического дифференцирования
  with tf.GradientTape() as g1:
    pred = neural_net(input_x) 
    #print(pred.shape)   
    loss = cross_entropy(pred, output_y)        
    global dw1
    global db1
    global dw2
    global db2
    global dw3
    global db3    
    dw1, db1, dw2, db2, dw3, db3 = g1.gradient(loss, [neural_net.layer_1.w, neural_net.layer_1.b, neural_net.layer_2.w, neural_net.layer_2.b, neural_net.layer_3.w, neural_net.layer_3.b])
    optimizer.apply_gradients(zip((dw1, db1, dw2, db2, dw3, db3), [neural_net.layer_1.w, neural_net.layer_1.b, neural_net.layer_2.w, neural_net.layer_2.b, neural_net.layer_3.w, neural_net.layer_3.b]))

loss_history = []  # каждые display_step шагов сохраняйте в этом список текущую ошибку нейросети
accuracy_history = [] # каждые display_step шагов сохраняйте в этом список текущую точность нейросети

# В этом цикле мы будем производить обучение нейронной сети
# из тренировочного датасета train_data извлеките случайное подмножество, на котором 
# произведется тренировка. Используйте метод take, доступный для тренировочного датасета.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1): # Место для вашего кода:
    # Обновляем веса нейронной сети
    # Место для вашего кода
    train(batch_x, batch_y)
    loss = cross_entropy(y_pred=neural_net(batch_x), y_true=batch_y)
    #optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if step % display_step == 0:
        pred = neural_net(batch_x)
        loss_history.append(loss.numpy())
        acc_num = accuracy(pred, batch_y)
        accuracy_history.append(acc_num)
        # Место для вашего кода    

print(loss_history)   
print(accuracy_history)

import matplotlib.pyplot as plt
n_images = 5
test_images = x_test[:n_images]
predictions = neural_net(test_images)

# display
print(predictions.numpy())
for i in range(n_images):
    plt.imshow(np.reshape(test_images[i],[28,28]),cmap='gray')
    plt.show()
    print('Model prediction:%i'%np.argmax(predictions.numpy()[i]))