# Reynolds Flocking Agents

Matplotlib animations of Reynolds-style boids with boundary steering, plus optional collision-avoidance variants.

## Run it
- Install deps: `pip install matplotlib numpy`
- Base flocking demo (wander/flock + bounce + boundary steer): `python agents.py`
- Flocking with optional avoidance (set `AVOIDANCE_MODE` to `TTC`, `RVO`, or `NONE`): `python agents_ttc.py`

## Behavior notes
- Agents spawn on a small grid, all facing +x; each step applies flocking (cohesion, separation, alignment) and gentle boundary steering.
- `agents.py` matches classic boids (with wander toggle) and bounce logic to stay in bounds.
- `agents_ttc.py` reuses the same agent model and adds two avoidance helpers:
  - `TTC`: time-to-collision steering away from predicted close passes inside a horizon.
  - `RVO`: reciprocal velocity obstacle-style nudging when closing in.
- Tune parameters in `Agent` (e.g., `perception_radius`, `max_turn`, `turn_rate`, speeds) or switch avoidance modes to explore different dynamics.
