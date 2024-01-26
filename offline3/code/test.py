# import numpy as np
# from scipy.stats import multivariate_normal
#
# # Import necessary libraries
# from sklearn import datasets  # to retrieve the iris Dataset
# import pandas as pd  # to load the dataframe
# from sklearn.preprocessing import StandardScaler  # to standardize the features
# from sklearn.decomposition import PCA  # to apply PCA
# import seaborn as sns  # to plot the heat maps
#
# # delta = 0.25
# # x = np.arange(-2.0, 7.0, delta)
# # y = np.arange(-2.0, 7.0, delta)
# # X, Y = np.meshgrid(x, y)
# #
# # print(X)
# # print(Y)
# #
# # var = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
# # Z = np.array([[var.pdf([X[i][j], Y[i][j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])
# # print(Z)
# # # print(var.pdf([[1,0], [1,0], [1,0]]))
# #
# # i = 0
# # j = 0
# # print(var.pdf([X[i][j], Y[i][j]]))
#
# print([i for i in range(3, 7)])
#
# X = np.array([
#     [1, 2, 3, 4],
#     [1, 1, 3, 7],
#     [1, 2, 5, 4],
#     [1, 9, 2, 4],
#     [1, 7, 3, 5],
# ])
#
# # Applying PCA
# # Taking no. of Principal Components as 3
# pca = PCA(n_components=2)
# pca.fit(X)
# data_pca = pca.transform(X)
#
# print(data_pca)


# importing libraries
import numpy as np
import time
import matplotlib.pyplot as plt

# creating initial data values
# of x and y
x = np.linspace(0, 10, 100)
y = np.sin(x)

# to run GUI event loop
plt.ion()

# here we are creating sub plots
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(x, y)

# setting title
plt.title("Geeks For Geeks", fontsize=20)

# setting x-axis label and y-axis label
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Loop
for _ in range(50):
    # creating new Y values
    new_y = np.sin(x - 0.5 * _)
    # ax = plt.gca()

    # updating data values
    line1.set_xdata(x)
    line1.set_ydata(new_y)

    plt.xticks(default_x_ticks, x)
    plt.show()

    # drawing updated values
    figure.canvas.draw()

    # This will run the GUI event
    # loop until all UI events
    # currently waiting have been processed
    figure.canvas.flush_events()

    time.sleep(0.1)
