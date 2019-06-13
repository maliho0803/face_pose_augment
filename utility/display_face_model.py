
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def DrawSolidHead(vertex, tri):
    fig = plt.figure()
    ax = fig.gca(projection='3d', azim=90, elev=90)
    ax.plot_trisurf(vertex[0, :], vertex[1, :], vertex[2, :], triangles=tri)
    plt.show()