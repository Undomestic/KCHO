"""
Visualisation and helper functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
import config

def plot_raw_vs_smoothed(raw, smoothed, show=True):
    fig = plt.figure(figsize=(14, 8))
    for i, var in enumerate(config.PHYSIO_VARS):
        plt.subplot(2, 3, i+1)
        plt.plot(raw['time'], raw[var], alpha=0.3, label='raw')
        plt.plot(smoothed['time'], smoothed[var], 'r-', label='Chebyshev')
        plt.title(var)
        plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_persistence(diagrams, save=False, show=True, ax=None):
    plot_diagrams(diagrams, show=show, ax=ax)
    if save:
        plt.savefig('persistence.png')

def plot_kan_edge(x, y, spline, x_grid, y_spline, var_in, var_out):
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, alpha=0.5, label='data')
    plt.plot(x_grid, y_spline, 'r-', label='learned φ')
    plt.xlabel(var_in)
    plt.ylabel(var_out)
    plt.title(f'KAN edge: {var_in} → {var_out}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_3d_point_cloud(point_cloud, color_by=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    if color_by is None:
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
    else:
        ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c=color_by, cmap='viridis')
    ax.set_xlabel('Dim1')
    ax.set_ylabel('Dim2')
    ax.set_zlabel('Dim3')
    plt.title('3D projection of physiological phase space')
    plt.show()

def plot_curvature_over_time(curvatures):
    plt.figure(figsize=(10,4))
    plt.plot(curvatures, 'b-')
    plt.xlabel('Time index')
    plt.ylabel('Ricci curvature')
    plt.title('Evolution of scalar curvature')
    plt.grid(True)
    plt.show()

def print_report(report_text):
    print(report_text)