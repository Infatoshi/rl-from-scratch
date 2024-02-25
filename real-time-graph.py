import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Initialize a list to hold your data, this will store the latest 200 points
data = []
max_length = 200  # Set the maximum number of points to display at a time

fig, ax = plt.subplots()
line, = ax.plot(data)

def update(frame):
    # Add new data (replace np.random.rand() with your actual data source)
    data.append(np.random.rand())
    
    # Ensure data only keeps the last 200 points
    if len(data) > max_length:
        data.pop(0)  # Remove the oldest data point
    
    # Update the data of your line with the new list
    line.set_ydata(data)
    # Update the x-axis data to match the new data's index
    line.set_xdata(range(len(data)))
    
    # Adjust the plot limits dynamically
    ax.set_xlim(0, max_length)
    ax.set_ylim(min(data), max(data))  # Optionally, adjust y-axis limits dynamically
    
    return line,

ani = FuncAnimation(fig, update, interval=10)

plt.show()
