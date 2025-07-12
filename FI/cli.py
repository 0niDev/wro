import numpy as np
import matplotlib.pyplot as plt

# Link lengths
link1 = 60
link2 = 40  # Total reach = 100

def solve_ik(x, y, z):
    theta1 = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(r_xy**2 + z**2)

    if r == 0 or r > (link1 + link2):
        print("❌ Out of range (0 < reach ≤ 100)")
        return

    cos_theta3 = (r**2 - link1**2 - link2**2) / (2 * link1 * link2)
    if abs(cos_theta3) > 1:
        print("❌ Target not reachable")
        return

    theta3 = -np.arccos(cos_theta3)
    k1 = link1 + link2 * np.cos(theta3)
    k2 = link2 * np.sin(theta3)
    theta2 = np.arctan2(z, r_xy) - np.arctan2(k2, k1)

    # Degrees
    theta1_deg = np.degrees(theta1)
    theta2_deg = np.degrees(theta2)
    theta3_deg = np.degrees(theta3)

    print(f"Theta1 (Base):     {theta1_deg:.2f}°")
    print(f"Theta2 (Shoulder): {theta2_deg:.2f}°")
    print(f"Theta3 (Elbow):    {theta3_deg:.2f}°")

    # Forward kinematics
    x1 = link1 * np.cos(theta2) * np.cos(theta1)
    y1 = link1 * np.cos(theta2) * np.sin(theta1)
    z1 = link1 * np.sin(theta2)

    x2 = x1 + link2 * np.cos(theta2 + theta3) * np.cos(theta1)
    y2 = y1 + link2 * np.cos(theta2 + theta3) * np.sin(theta1)
    z2 = z1 + link2 * np.sin(theta2 + theta3)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([0, x1], [0, y1], [0, z1], 'b-', linewidth=4, label='Link 1')
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'g-', linewidth=4, label='Link 2')
    ax.scatter([0, x1, x2], [0, y1, y2], [0, z1, z2], c='k', s=40)
    ax.scatter(x, y, z, c='r', s=50, label='Target')
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_zlim(0, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3DOF Arm (Yaw + XZ Planar)")
    ax.legend()

    # Wait for key press
    print("Press Q to enter new values...")
    def on_key(event):
        if event.key.lower() == 'q':
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


# 🔁 Repeat loop
while True:
    try:
        x = float(input("Enter X: "))
        y = float(input("Enter Y: "))
        z = float(input("Enter Z: "))
        solve_ik(x, y, z)
    except:
        print("❌ Invalid input")
