import matplotlib.pyplot as plt

def plot_trajectory(x, y, title: str):
    plt.scatter(x, y, color='blue')
    plt.title(title)
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.xlim(right=1920) 
    plt.ylim(top=1080)
    # plt.gca().set_aspect('equal')  # Set the same scale for both axes
    plt.gca().invert_yaxis()  # The origin is at the top-left corner
    plt.grid(True)