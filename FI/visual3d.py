import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Link lengths
link1 = 3
link2 = 2

# Setup 3D figure and sliders
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3DOF Robot Arm (Base + Shoulder + Elbow)")
ax.grid(True)

# Plot handles
link1_line, = ax.plot([], [], [], 'b-', linewidth=3, label='Link 1')
link2_line, = ax.plot([], [], [], 'g-', linewidth=3, label='Link 2')
joint_dots = ax.scatter([], [], [], c='k', s=40)
target_dot = ax.scatter([], [], [], c='r', s=40, label='Target')

# Sliders
ax_x = plt.axes([0.25, 0.18, 0.65, 0.03])
ax_y = plt.axes([0.25, 0.13, 0.65, 0.03])
ax_z = plt.axes([0.25, 0.08, 0.65, 0.03])
slider_x = Slider(ax_x, 'X', -4, 4, valinit=3)
slider_y = Slider(ax_y, 'Y', -4, 4, valinit=0)
slider_z = Slider(ax_z, 'Z', 0, 4, valinit=2)

def update(val):
    x = slider_x.val
    y = slider_y.val
    z = slider_z.val

    # Base rotation
    theta1 = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(r_xy**2 + z**2)

    if r == 0 or r > (link1 + link2):
        print("❌ Out of range")
        link1_line.set_data([], [])
        link1_line.set_3d_properties([])
        link2_line.set_data([], [])
        link2_line.set_3d_properties([])
        joint_dots._offsets3d = ([], [], [])
        target_dot._offsets3d = ([x], [y], [z])
        fig.canvas.draw_idle()
        return

    # Inverse kinematics (elbow-down)
    cos_theta3 = (r**2 - link1**2 - link2**2) / (2 * link1 * link2)
    if abs(cos_theta3) > 1:
        print("❌ IK unreachable")
        return

    theta3 = -np.arccos(cos_theta3)
    k1 = link1 + link2 * np.cos(theta3)
    k2 = link2 * np.sin(theta3)
    theta2 = np.arctan2(z, r_xy) - np.arctan2(k2, k1)

    # Forward kinematics
    x1 = link1 * np.cos(theta2) * np.cos(theta1)
    y1 = link1 * np.cos(theta2) * np.sin(theta1)
    z1 = link1 * np.sin(theta2)

    x2 = x1 + link2 * np.cos(theta2 + theta3) * np.cos(theta1)
    y2 = y1 + link2 * np.cos(theta2 + theta3) * np.sin(theta1)
    z2 = z1 + link2 * np.sin(theta2 + theta3)

    # Plot
    link1_line.set_data([0, x1], [0, y1])
    link1_line.set_3d_properties([0, z1])
    link2_line.set_data([x1, x2], [y1, y2])
    link2_line.set_3d_properties([z1, z2])
    joint_dots._offsets3d = ([0, x1, x2], [0, y1, y2], [0, z1, z2])
    target_dot._offsets3d = ([x], [y], [z])

    fig.canvas.draw_idle()

# Connect sliders
slider_x.on_changed(update)
slider_y.on_changed(update)
slider_z.on_changed(update)
update(None)
plt.legend()
plt.show()
