import matplotlib.pyplot as plt

# constants
plot_title = "Q-Learning"
x_axis_name = "Episode"
y_axis_name = "Target Reached"

# global variables
x_coords = []
y_coords = []

def init():
    global figure, line, axis
    plt.ion()
    figure, axis = plt.subplots()
    line, = axis.plot(x_coords, y_coords)
    plt.title(plot_title, fontsize=15)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)

def update_plot():
    line.set_xdata(x_coords)
    line.set_ydata(y_coords)
    axis.relim()
    axis.autoscale_view()
    figure.canvas.draw()
    figure.canvas.flush_events()

def keep_plot():
    while True:
        pass