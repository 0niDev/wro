import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sympy as sp

# Link lengths
link1 = 3
link2 = 2

# Setup figure and sliders
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_xlim(-2, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True)
base_line, = ax.plot([-2, 5], [0, 0], 'k-')

# Plot handles
link1_line, = ax.plot([], [], 'b-', linewidth=3)
link2_line, = ax.plot([], [], 'g-', linewidth=3)
joint1_dot, = ax.plot([], [], 'bo', markersize=10)
joint2_dot, = ax.plot([], [], 'go', markersize=10)
effector_dot, = ax.plot([], [], 'r.', markersize=20)
gripper_lines = []

# Sliders
ax_x = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_y = plt.axes([0.2, 0.05, 0.65, 0.03])
slider_x = Slider(ax_x, 'Target X', -1.0, 5.0, valinit=4)
slider_y = Slider(ax_y, 'Target Y', -1.0, 5.0, valinit=2)

def draw_gripper(x2, y2, x3, y3):
    for g in gripper_lines:
        g.remove()
    gripper_lines.clear()

    x, y = sp.symbols('x y')
    if x3 == x2:
        g1, = ax.plot([x2 - 0.3, x2 + 0.3], [y2, y2], 'r-', linewidth=3)
        g2, = ax.plot([x2 - 0.3, x2 - 0.3], [y2, y2 + 0.1], 'r-', linewidth=3)
        g3, = ax.plot([x2 + 0.3, x2 + 0.3], [y2, y2 + 0.1], 'r-', linewidth=3)
        gripper_lines.extend([g1, g2, g3])
    else:
        m1 = (y3 - y2) / (x3 - x2)
        if m1 == 0:
            g1, = ax.plot([x2, x2], [y2 - 0.3, y2 + 0.3], 'r-', linewidth=3)
            g2, = ax.plot([x2, x2 + 0.1], [y2 + 0.3, y2 + 0.3], 'r-', linewidth=3)
            g3, = ax.plot([x2, x2 + 0.1], [y2 - 0.3, y2 - 0.3], 'r-', linewidth=3)
            gripper_lines.extend([g1, g2, g3])
        else:
            m2 = -1 / m1
            c = y2 - m2 * x2
            func1 = y - m2 * x - c
            func2 = (x - x2)**2 + (y - y3)**2 - 0.3**2
            sol = sp.solve([func1, func2], (x, y))

            if len(sol) == 2:
                x1_, y1_ = float(sol[0][0]), float(sol[0][1])
                x2_, y2_ = float(sol[1][0]), float(sol[1][1])
                g1, = ax.plot([x1_, x2_], [y1_, y2_], 'r-', linewidth=3)
                g2, = ax.plot([x1_, x1_ + 300 * (x3 - x2)], [y1_, y1_ + 300 * (y3 - y2)], 'r-', linewidth=3)
                g3, = ax.plot([x2_, x2_ + 300 * (x3 - x2)], [y2_, y2_ + 300 * (y3 - y2)], 'r-', linewidth=3)
                gripper_lines.extend([g1, g2, g3])

def update(val):
    target_x = slider_x.val
    target_y = slider_y.val

    # Inverse kinematics
    cos_theta2 = (target_x**2 + target_y**2 - link1**2 - link2**2) / (2 * link1 * link2)
    if abs(cos_theta2) > 1:
        print("❌ Target out of reach")
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        joint1_dot.set_data([], [])
        joint2_dot.set_data([], [])
        effector_dot.set_data([], [])
        for g in gripper_lines:
            g.remove()
        gripper_lines.clear()
        fig.canvas.draw_idle()
        return

    theta2 = -np.arccos(cos_theta2)  # elbow-down
    k1 = link1 + link2 * np.cos(theta2)
    k2 = link2 * np.sin(theta2)
    theta1 = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)

    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = 0

    # Forward kinematics
    x1 = link1 * np.cos(np.radians(theta1_deg))
    y1 = link1 * np.sin(np.radians(theta1_deg))
    x2 = x1 + link2 * np.cos(np.radians(theta1_deg + theta2_deg))
    y2 = y1 + link2 * np.sin(np.radians(theta1_deg + theta2_deg))
    x3 = x2 + 0.001 * np.cos(np.radians(theta1_deg + theta2_deg + theta3_deg))
    y3 = y2 + 0.001 * np.sin(np.radians(theta1_deg + theta2_deg + theta3_deg))

    joint1_dot.set_data([0], [0])
    joint2_dot.set_data([x1], [y1])
    link1_line.set_data([0, x1], [0, y1])
    link2_line.set_data([x1, x2], [y1, y2])
    effector_dot.set_data([x3], [y3])
    draw_gripper(x2, y2, x3, y3)
    fig.canvas.draw_idle()

slider_x.on_changed(update)
slider_y.on_changed(update)
update(None)
plt.show()
