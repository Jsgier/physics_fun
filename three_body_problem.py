import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def three_body_equations(t, Y, G, m1, m2, m3):
    """
    Defines the system of differential equations for the three-body problem.

    Parameters:
        t: float, current time.
        Y: array_like, current values of the positions and velocities.
        G: float, gravitational constant.
        m1, m2, m3: float, masses of the bodies.

    Returns:
        dYdt: array_like, derivatives of positions and velocities.
    """
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, x3, y3, z3, vx3, vy3, vz3 = Y

    # Distance between bodies
    r12 = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    r13 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)
    r23 = np.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)

    # Derivatives
    dx1dt = vx1
    dy1dt = vy1
    dz1dt = vz1
    dvx1dt = G * ((m2 * (x2 - x1) / r12**3) + (m3 * (x3 - x1) / r13**3))
    dvy1dt = G * ((m2 * (y2 - y1) / r12**3) + (m3 * (y3 - y1) / r13**3))
    dvz1dt = G * ((m2 * (z2 - z1) / r12**3) + (m3 * (z3 - z1) / r13**3))

    dx2dt = vx2
    dy2dt = vy2
    dz2dt = vz2
    dvx2dt = G * ((m1 * (x1 - x2) / r12**3) + (m3 * (x3 - x2) / r23**3))
    dvy2dt = G * ((m1 * (y1 - y2) / r12**3) + (m3 * (y3 - y2) / r23**3))
    dvz2dt = G * ((m1 * (z1 - z2) / r12**3) + (m3 * (z3 - z2) / r23**3))

    dx3dt = vx3
    dy3dt = vy3
    dz3dt = vz3
    dvx3dt = G * ((m1 * (x1 - x3) / r13**3) + (m2 * (x2 - x3) / r23**3))
    dvy3dt = G * ((m1 * (y1 - y3) / r13**3) + (m2 * (y2 - y3) / r23**3))
    dvz3dt = G * ((m1 * (z1 - z3) / r13**3) + (m2 * (z2 - z3) / r23**3))

    dYdt = [dx1dt, dy1dt, dz1dt, dvx1dt, dvy1dt, dvz1dt,
            dx2dt, dy2dt, dz2dt, dvx2dt, dvy2dt, dvz2dt,
            dx3dt, dy3dt, dz3dt, dvx3dt, dvy3dt, dvz3dt]

    return dYdt

# Initial conditions
G = 6.67430e-11  # Gravitational constant
m1 = 1.0e25      # Mass of body 1
m2 = 1.5e25      # Mass of body 2
m3 = 2.0e26      # Mass of body 3
Y0 = [0, 0, 5, 0, 0, 0,    # Initial position and velocity of body 1
      10, 0, 0, 0, 7, 0,    # Initial position and velocity of body 2
      0, 10, 0, -5, -5, 0]  # Initial position and velocity of body 3

# Time span
t_span = (0, 100)  # Time interval for integration

# Integrate the equations of motion
sol = solve_ivp(lambda t, Y: three_body_equations(t, Y, G, m1, m2, m3), t_span, Y0, method='RK45')

# Create a larger figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initialize the lines for each body
lines = [ax.plot([], [], [], label=f'Body {i+1}')[0] for i in range(3)]

# Function to update the plot
def update_plot(i, lines, sol):
    for j, line in enumerate(lines):
        line.set_data(sol.y[j*6, :i], sol.y[j*6+1, :i])
        line.set_3d_properties(sol.y[j*6+2, :i])
    return lines

# Animate the plot
ani = FuncAnimation(fig, update_plot, frames=sol.y.shape[1], fargs=(lines, sol), interval=50, blit=True)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectories of Three Bodies')
ax.legend()

# Set the limits of the x, y, and z axes
margin = 0  # Margin around the trajectory
ax.set_xlim(min(sol.y[::6].min(), sol.y[1::6].min(), sol.y[2::6].min()) - margin,
            max(sol.y[::6].max(), sol.y[1::6].max(), sol.y[2::6].max()) + margin)
ax.set_ylim(min(sol.y[1::6].min(), sol.y[1::6].min(), sol.y[2::6].min()) - margin,
            max(sol.y[::6].max(), sol.y[1::6].max(), sol.y[2::6].max()) + margin)
ax.set_zlim(min(sol.y[2::6].min(), sol.y[1::6].min(), sol.y[2::6].min()) - margin,
            max(sol.y[::6].max(), sol.y[1::6].max(), sol.y[2::6].max()) + margin)

# Show the animation
plt.show()
