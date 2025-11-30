"""Reynolds-style agents with optional TTC or RVO avoidance on top of the
Agent implementation from agents.py."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import random

# --- Simulation configuration ---
AVOIDANCE_MODE = "RVO"  # options: "TTC", "RVO", "NONE"
TIME_HORIZON = 5.0


class Agent:
    def __init__(
        self,
        ax,
        x=0.0,
        y=0.0,
        angle=0.0,
        v=0.02,
        omega=0.0,
        radius=0.25,
        facecolor="C1",
        edgecolor="k",
    ):
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
        self.perception_radius = 6.5  # larger neighborhood for better alignment
        self.max_turn = 7.5  # more responsive steering
        self.max_speed = 0.75  # slower top speed to keep formations coherent
        self.min_speed = 0.2
        self.separation_distance = radius * 2.8
        self.separation_gain = 1.3

        # Collision avoidance
        self.collision_radius = radius * 3.0

        self.circle = Circle(
            (x, y),
            radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=0.6,
            lw=1.5,
            zorder=2,
        )
        ax.add_patch(self.circle)
        (self.orientation_line,) = ax.plot([], [], color=edgecolor, lw=2, zorder=3)
        (self.head_marker,) = ax.plot([], [], "o", color=edgecolor, markersize=5, zorder=4)

    def velocity_vec(self):
        theta = np.deg2rad(self.angle)
        return np.array([np.cos(theta) * self.v, np.sin(theta) * self.v])

    def set_command(self, v, omega):
        self.v = v
        self.omega = omega

    def wander(self):
        if random.random() < 0.02:
            self.target_angle = (self.angle + random.uniform(-90, 90)) % 360

        diff = (self.target_angle - self.angle + 540) % 360 - 180
        self.omega = np.clip(diff, -self.turn_rate, self.turn_rate)
        self.v = max(0.0, self.v_mean + random.uniform(-0.005, 0.005))

    def _ttc_avoidance(self, neighbors, time_horizon):
        """Time-to-collision avoidance vector."""
        avoidance = np.zeros(2)
        v_i = self.velocity_vec()
        for n in neighbors:
            r = np.array([n.x - self.x, n.y - self.y])
            v_rel = v_i - n.velocity_vec()
            vrel2 = np.dot(v_rel, v_rel)
            if vrel2 < 1e-9:
                continue
            t_star = -np.dot(r, v_rel) / vrel2
            if 0 < t_star < time_horizon:
                closest = r + v_rel * t_star
                d2 = np.dot(closest, closest)
                if d2 < self.collision_radius**2:
                    dir_away = -closest / (np.linalg.norm(closest) + 1e-9)
                    weight = (self.collision_radius - np.sqrt(d2)) / self.collision_radius
                    weight *= 1.0 / (t_star + 1e-3)
                    avoidance += dir_away * weight
        return avoidance

    def _rvo_avoidance(self, neighbors):
        """Reciprocal velocity obstacle style steering."""
        avoidance = np.zeros(2)
        v_i = self.velocity_vec()
        for n in neighbors:
            r = np.array([n.x - self.x, n.y - self.y])
            dist = np.linalg.norm(r)
            if dist < 1e-9:
                continue
            v_rel = v_i - n.velocity_vec()
            if dist < self.collision_radius:
                avoidance += (r / dist) * 0.5
            elif np.dot(v_rel, r) < 0:  # closing in
                avoidance += (r / dist) * 0.25
        return avoidance

    def flock(self, neighbors, avoidance_mode="NONE", time_horizon=1.5):
        """Classic Reynolds boids plus optional TTC/RVO avoidance."""
         # If boundary steering is active, do not modify ω
        if getattr(self, "boundary_override", False):
            return
        if not neighbors:
            return

        positions = np.array([[n.x, n.y] for n in neighbors])
        velocities = np.array(
            [
                [np.cos(np.deg2rad(n.angle)) * n.v, np.sin(np.deg2rad(n.angle)) * n.v]
                for n in neighbors
            ]
        )

        self_pos = np.array([self.x, self.y])

        # Cohesion
        center_of_mass = positions.mean(axis=0)
        cohesion_vec = (center_of_mass - self_pos) * 0.05

        # Separation
        separation_vec = np.zeros(2)
        for n in neighbors:
            dx, dy = self.x - n.x, self.y - n.y
            dist = np.hypot(dx, dy)
            if dist < self.separation_distance and dist > 1e-6:
                separation_vec += (np.array([dx, dy]) / dist**2) * self.separation_gain

        # Alignment
        avg_vel = velocities.mean(axis=0)
        alignment_vec = (
            avg_vel
            - np.array([np.cos(np.deg2rad(self.angle)) * self.v, np.sin(np.deg2rad(self.angle)) * self.v])
        ) * 0.18

        # Optional collision avoidance
        avoidance_vec = np.zeros(2)
        if avoidance_mode.upper() == "TTC":
            avoidance_vec = self._ttc_avoidance(neighbors, time_horizon)
        elif avoidance_mode.upper() == "RVO":
            avoidance_vec = self._rvo_avoidance(neighbors)

        steering = cohesion_vec + separation_vec + alignment_vec + avoidance_vec
        norm = np.linalg.norm(steering)
        if norm < 1e-9:
            return

        desired_angle = np.rad2deg(np.arctan2(steering[1], steering[0])) % 360
        diff = (desired_angle - self.angle + 540) % 360 - 180
        self.omega = np.clip(diff, -self.max_turn, self.max_turn)

        self.v = np.clip(self.v, self.min_speed, self.max_speed)

    def bound_steer(self, xmin, xmax, ymin, ymax,
                buffer=2.0, strength=0.05):
        steer = np.zeros(2)

        # Check distances
        dx_min = self.x - xmin
        dx_max = xmax - self.x
        dy_min = self.y - ymin
        dy_max = ymax - self.y

        # Build steering vector
        if dx_min < buffer:
            steer[0] += strength * (buffer - dx_min)
        if dx_max < buffer:
            steer[0] -= strength * (buffer - dx_max)
        if dy_min < buffer:
            steer[1] += strength * (buffer - dy_min)
        if dy_max < buffer:
            steer[1] -= strength * (buffer - dy_max)

        # If no boundary influence → normal flocking allowed
        if np.linalg.norm(steer) < 1e-6:
            self.boundary_override = False
            return False

        # Otherwise we ARE in boundary steering mode
        self.boundary_override = True

        # Convert vector to desired heading
        desired_angle = np.rad2deg(np.arctan2(steer[1], steer[0])) % 360
        diff = (desired_angle - self.angle + 540) % 360 - 180

        # Strong clamp so boundary always wins
        self.omega = np.clip(diff, -self.max_turn, self.max_turn)

        return True


    def update(self, xmin, xmax, ymin, ymax):
        self.angle = (self.angle + self.omega) % 360
        theta = np.deg2rad(self.angle)

        # Reynolds-style soft boundaries
        self.bound_steer(xmin, xmax, ymin, ymax)
        
        # # Bounce from walls
        # if self.x - self.radius < xmin or self.x + self.radius > xmax:
        #     self.x = np.clip(self.x, xmin + self.radius, xmax - self.radius)
        #     self.angle = (180 - self.angle) % 360
        #     self.target_angle = self.angle
        #     theta = np.deg2rad(self.angle)
        # if self.y - self.radius < ymin or self.y + self.radius > ymax:
        #     self.y = np.clip(self.y, ymin + self.radius, ymax - self.radius)
        #     self.angle = (-self.angle) % 360
        #     self.target_angle = self.angle
        #     theta = np.deg2rad(self.angle)


        self.x += self.v * np.cos(theta)
        self.y += self.v * np.sin(theta)

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

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_aspect("equal")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Reynolds Agents ({AVOIDANCE_MODE} avoidance)")
ax.grid(True)

agents = []
n_agents = 6
n_cols = 2
spacing = 0.5
initial_angle = 90.0

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
    for a in agents:
        neighbors = [
            n
            for n in agents
            if n is not a and np.hypot(n.x - a.x, n.y - a.y) < a.perception_radius
        ]
        a.flock(neighbors, avoidance_mode=AVOIDANCE_MODE, time_horizon=TIME_HORIZON)
        artists.extend(a.update(xmin, xmax, ymin, ymax))
    return artists


anim = FuncAnimation(fig, update, init_func=init, frames=400, interval=30, blit=True)
plt.show()
