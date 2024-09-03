import matplotlib.pyplot as plt
import numpy as np  
# from mpl_toolkits.mplot3d import Axes3D


# Data's here can be changed, according to requirements
data = [
    [10, 15, 20, 25, 30], 
    [120, 5800, 2100, 600, 3200],  
    [340, 410, 320, 470, 530],   
    [6000, 4500, 905, 7088, 2100],
    [1010, 98, 35, 510, 325]
]

# convert the list of lists to a NumPy array
data = np.array(data)  # This ensures proper handling by plot_surface

# create the figure and axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set up grid
x, y = np.meshgrid(np.arange(len(data[0])), np.arange(len(data)))

# create the waterfall plot
ax.plot_surface(x, y, data, cmap='viridis', edgecolors='black', alpha=0.7)

# customize the plot
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Waterfall Plot')

# show
plt.show()


