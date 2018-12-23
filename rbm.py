import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


class RBM:
    """Restricted Boltzmann Machine."""

    def __init__(self, n_hidden=2, n_observe=784):
        """Initialize model."""
        self.n_hidden = n_hidden
        self.n_observe = n_observe
        self.batch_size = 32
        self.learning_rate = 0.01
        self.max_epoch = 200
        self.W = np.zeros([self.n_observe, self.n_hidden])
        self.a = np.zeros([self.n_observe])
        self.b = np.zeros([self.n_hidden])

    def train(self, data):
        """Train model using data."""

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def energy(v0, h0):
            return np.mean(-np.matmul(v0, self.a)
                           - np.matmul(h0, self.b)
                           - np.matmul(
                np.matmul(h0, np.transpose(self.W)),
                np.transpose(v0)))

        n = mnist.train.images.shape[0]

        for epoch in range(self.max_epoch):
            for i in range(int(n / self.batch_size)):
                batch_x, batch_y = data.train.next_batch(self.batch_size)

                v = np.around(batch_x)

                h = np.random.binomial(
                    1, sigmoid(np.add(self.b, np.dot(v, self.W))))
                forward_gradient = np.dot(np.transpose(v), h)

                v_1 = np.random.binomial(
                    1,
                    sigmoid(np.add(self.a, np.dot(h, np.transpose(self.W)))))
                h_1 = np.random.binomial(
                    1, sigmoid(np.add(self.b, np.dot(v_1, self.W))))

                backward_gradient = np.dot(np.transpose(v_1), h_1)

                self.W = self.W + (self.learning_rate *
                                   (forward_gradient - backward_gradient) / self.batch_size)
                self.a = (self.a + self.learning_rate * np.mean(v - v_1, 0))
                self.b = (self.b + self.learning_rate * np.mean(h - h_1, 0))

            loss = np.mean(np.sum(np.square(v - v_1), axis=1))
            print('Epoch %d/%d: Energy: %f, Loss: %f' %
                  (epoch + 1, self.max_epoch, energy(v, h_1), loss))
            if not epoch % 20:
                np.save("w.npy", self.W)
                np.save("a.npy", self.a)
                np.save("b.npy", self.b)

    def sample(self, p_v):
        """Sample from trained model."""

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        for i in range(3):
            p_h = sigmoid(np.add(self.b, np.dot(p_v, self.W)))
            p_v = sigmoid(np.add(self.a, np.dot(p_h, np.transpose(self.W))))
        return p_v


# train restricted boltzmann machine using mnist dataset
if __name__ == '__main__':
    # load mnist dataset, no label
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    n_imgs, img_size = mnist.train.images.shape
    print(mnist.train.images.shape)

    # construct rbm model
    rbm = RBM(100, img_size)

    # train rbm model using mnist
    rbm.train(mnist)

    # sample from rbm model
    for j in range(20):
        test_image = mnist.test.images[j].reshape(-1, 1)
        s = rbm.sample(test_image)
        plt.subplot(1, 2, 1)
        plt.imshow(test_image.reshape((28, 28)), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(s.reshape((28, 28)), cmap="gray")
        plt.savefig('result' + str(j + 1) + '.png')
