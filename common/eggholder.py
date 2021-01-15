"""Generate a Complicated Surface."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))
          -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

def generateSurface(isPlot=True):
    x = np.arange(-512, 513)
    y = np.arange(-512, 513)
    xgrid, ygrid = np.meshgrid(x, y)
    xy = np.stack([xgrid, ygrid])
    
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.plot_surface(xgrid, ygrid, eggholder(xy), cmap='terrain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Optimization Landscape(x, y)')
    ax.set_title("Complex, Non-Convex Curve")
    if isPlot:
        plt.show()

if __name__ == '__main__':
    generateSurface(isPlot=False)
