# Reynolds Flocking Agents

Simple matplotlib animation showcasing Reynolds-style boid behaviors (cohesion, separation, alignment) with wall steering.

## Run it
- Install deps: `pip install matplotlib numpy`
- Start the animation: `python agents.py`

## Notes
- Agents spawn on a small grid, all facing +x; each step applies flocking (or wander) plus boundary steering.
- Walls are handled by gentle steering plus bounce logic to keep agents in bounds.
- Customize behavior by tweaking parameters in `Agent` (e.g., `perception_radius`, `max_turn`, `turn_rate`, speeds).
