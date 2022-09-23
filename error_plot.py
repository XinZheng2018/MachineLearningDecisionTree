import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    depth = np.array([0,1,2,3,4,5,6])
    train_error = np.array([0.4429530201342282, 0.20134228187919462, 0.1342281879194631, 0.11409395973154363, 0.10738255033557047, 0.087248322147651, 0.0738255033557047])
    test_error = np.array([0.5060240963855421, 0.21686746987951808,0.1566265060240964, 0.1686746987951807, 0.20481927710843373, 0.1686746987951807, 0.1927710843373494])

    plt.plot(depth, train_error, label='train_error')
    plt.plot(depth, test_error, label='test_error')
    leg = plt.legend(loc='upper center')

    plt.xlabel("max_depth")
    plt.ylabel("error")

    plt.show()