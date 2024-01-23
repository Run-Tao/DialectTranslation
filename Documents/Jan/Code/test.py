import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 10, 0.01)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
