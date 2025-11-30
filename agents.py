import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import random

class Agent:
    def __init__(self, ax, x=0.0, y=0.0, angle=0.0,
                 v=0.02, omega=0.0, radius=0.25,
                 facecolor='C1', edgecolor='k'):
        self.x = x
        self.y = y
        self.angle = angle  # degrees
        self.v = v
        self.omega = omega
        self.radius = radius

        # Wander params
        self.target_angle = angle
        self.turn_rate = 2.0  # max deg per frame
        self.v_mean = v

        # Flocking params
        self.perception_radius = 5.0
        self.max_turn = 5.0
        self.max_speed = 4.9
        self.min_speed = 0.1

        self.circle = Circle((x, y), radius,
                             facecolor=facecolor, edgecolor=edgecolor,
                             alpha=0.6, lw=1.5, zorder=2)
        ax.add_patch(self.circle)
        (self.orientation_line,) = ax.plot([], [], color=edgecolor, lw=2, zorder=3)
        (self.head_marker,) = ax.plot([], [], "o", color=edgecolor, markersize=5, zorder=4)

    def set_command(self, v, omega):
        self.v = v
        self.omega = omega

    def wander(self):
        if random.random() < 0.02:
            self.target_angle = (self.angle + random.uniform(-90, 90)) % 360

        diff = (self.target_angle - self.angle + 540) % 360 - 180
        self.omega = np.clip(diff, -self.turn_rate, self.turn_rate)
        self.v = max(0.0, self.v_mean + random.uniform(-0.005, 0.005))

    def flock(self, neighbors):
        """Classic Reynolds boids: cohesion, separation, alignment."""
        if not neighbors:
            # If no neighbors, keep previous heading (or wander if you prefer)
            return

        # -------------------------
        # Prepare arrays
        # -------------------------
        positions = np.array([[n.x, n.y] for n in neighbors])
        velocities = np.array([
            [np.cos(np.deg2rad(n.angle)) * n.v,
            np.sin(np.deg2rad(n.angle)) * n.v]
            for n in neighbors
        ])

        self_pos = np.array([self.x, self.y])

        # -------------------------
        # RULE 1: COHESION
        # Move 1% toward center of mass
        # -------------------------
        center_of_mass = positions.mean(axis=0)
        cohesion_vec = (center_of_mass - self_pos) * 0.05

        # -------------------------
        # RULE 2: SEPARATION
        # Strong repulsion within small radius
        # -------------------------
        separation_vec = np.zeros(2)
        for n in neighbors:
            dx, dy = self.x - n.x, self.y - n.y
            dist = np.hypot(dx, dy)
            if dist < 0.4 and dist > 1e-6:
                separation_vec += np.array([dx, dy]) / dist**2

        # -------------------------
        # RULE 3: ALIGNMENT
        # Match average velocity of neighbors
        # -------------------------
        avg_vel = velocities.mean(axis=0)
        # steer 12.5% toward matched velocity
        alignment_vec = (avg_vel - np.array([
            np.cos(np.deg2rad(self.angle)) * self.v,
            np.sin(np.deg2rad(self.angle)) * self.v
        ])) * 0.125

        # -------------------------
        # Combine rule influence
        # -------------------------
        steering = cohesion_vec + separation_vec + alignment_vec

        # Convert steering vector → desired angle
        desired_angle = np.rad2deg(np.arctan2(steering[1], steering[0])) % 360

        # Heading change (clamped)
        diff = (desired_angle - self.angle + 540) % 360 - 180
        self.omega = np.clip(diff, -self.max_turn, self.max_turn)

        # Speed nudging (optional)
        self.v = np.clip(self.v, self.min_speed, self.max_speed)

    def bound_steer(self, xmin, xmax, ymin, ymax, buffer=2.0, strength=0.05):
        """Reynolds-style boundary steering (no bouncing)."""
        steer = np.zeros(2)

        # Distance from each boundary
        dx_min = self.x - xmin
        dx_max = xmax - self.x
        dy_min = self.y - ymin
        dy_max = ymax - self.y

        # If inside buffer zone → add inward steering
        if dx_min < buffer:
            steer[0] += strength * (buffer - dx_min)
        if dx_max < buffer:
            steer[0] -= strength * (buffer - dx_max)
        if dy_min < buffer:
            steer[1] += strength * (buffer - dy_min)
        if dy_max < buffer:
            steer[1] -= strength * (buffer - dy_max)

        if np.linalg.norm(steer) < 1e-6:
            return None  # no steering needed

        # Convert vector to heading
        desired_angle = np.rad2deg(np.arctan2(steer[1], steer[0])) % 360
        diff = (desired_angle - self.angle + 540) % 360 - 180

        # Apply a gentle turn
        self.omega = np.clip(diff, -self.max_turn, self.max_turn)

        return True


    def update(self, xmin, xmax, ymin, ymax):
        self.angle = (self.angle + self.omega) % 360
        theta = np.deg2rad(self.angle)

        self.x += self.v * np.cos(theta)
        self.y += self.v * np.sin(theta)

        # Bounce from walls
        if self.x - self.radius < xmin or self.x + self.radius > xmax:
            self.x = np.clip(self.x, xmin + self.radius, xmax - self.radius)
            self.angle = (180 - self.angle) % 360
            self.target_angle = self.angle
            theta = np.deg2rad(self.angle)
        if self.y - self.radius < ymin or self.y + self.radius > ymax:
            self.y = np.clip(self.y, ymin + self.radius, ymax - self.radius)
            self.angle = (-self.angle) % 360
            self.target_angle = self.angle
            theta = np.deg2rad(self.angle)

        cx, cy = self.x, self.y
        x_end = cx + self.radius * np.cos(theta)
        y_end = cy + self.radius * np.sin(theta)

        self.circle.center = (cx, cy)
        self.orientation_line.set_data([cx, x_end], [cy, y_end])
        self.head_marker.set_data([x_end], [y_end])

        return self.circle, self.orientation_line, self.head_marker

# --- Simulation setup ---
xmin, xmax = -10.5, 10.5
ymin, ymax = -5, 5

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Reynolds Flocking Agents")
ax.grid(True)

# --- Agent generation: 10 agents in a grid, aligned headings ---
agents = []
n_agents = 5
n_cols = 1  # number of columns in the grid
spacing = 0.6  # spacing between agents
initial_angle = 90.0  # all agents face +x

# Compute rows/cols positions so they are centered in the arena
x_positions = np.linspace(-spacing, spacing, n_cols)
n_rows = (n_agents + n_cols - 1) // n_cols
y_positions = np.linspace(-spacing, spacing, n_rows)

count = 0
for y in y_positions:
    for x in x_positions:
        if count >= n_agents:
            break
        agents.append(Agent(ax, x=x, y=y, angle=initial_angle, facecolor=f"C{count % 10}"))
        count += 1


def init():
    artists = []
    for a in agents:
        artists.extend(a.update(xmin, xmax, ymin, ymax))
    return artists

def update(frame):
    artists = []
    flocking_mode = True

    for a in agents:
        if flocking_mode:
            neighbors = [n for n in agents
                         if n is not a and np.hypot(n.x - a.x, n.y - a.y) < a.perception_radius]
            a.flock(neighbors)
        else:
            a.wander()
        artists.extend(a.update(xmin, xmax, ymin, ymax))
    return artists

anim = FuncAnimation(fig, update, init_func=init,
                     frames=400, interval=30, blit=True)
plt.show()