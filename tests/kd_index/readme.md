# KDIndex Test Scenarios – Visual Cheat Sheet

This note summarizes the **three unit-test scenarios** for `KDIndex`, with tiny, self-contained grids and the exact arithmetic used to decide the winning `tile_id`. 

---

## Notation (shared)
- Projection used by `KDIndex`: **x = lon · cos(lat₀), y = lat** (degrees).
- Distance compared: **Euclidean** in projected degrees.
- `lat₀` = mean latitude of the sea points in each scenario.
- Row‑major `tile_id_map` (C‑order) over **sea** cells.

---

## 1) `test_same_lon_closest_lat`
**Grid**
- `lons = [0, 1, 2]`, `lats = [10, 11, 12]`, **all sea**.

**tile_id_map**
```
[[0, 1, 2],
 [3, 4, 5],
 [6, 7, 8]]
```
**Sea coords by tile_id**
- 0:(0,10) 1:(1,10) 2:(2,10) 3:(0,11) 4:(1,11) 5:(2,11) 6:(0,12) 7:(1,12) 8:(2,12)

**Reference latitude**
- `lat₀ = mean(lats) = 11°` → `cos(lat₀) ≈ 0.981627`.

**Query**
- `q = (lon=1.0, lat=10.2)` (same **lon** as tiles 1 and 4).

**Projected coordinates**
- `x(q) = 1.0 · 0.981627 = 0.981627`, `y(q) = 10.2`.
- Tile 1: `(x=0.981627, y=10)` → Δx=0, Δy=0.2 → **d=0.2**.
- Tile 4: `(x=0.981627, y=11)` → Δx=0, Δy=0.8 → **d=0.8**.

**Chosen tile**: **tile_id = 1** (closest latitude when longitude is equal).

---

## 2) `test_same_lat_closest_lon`
**Grid & tile_id_map**: same as (1).

**Reference latitude**
- `lat₀ = 11°` → `cos(lat₀) ≈ 0.981627`.

**Query**
- `q = (lon=0.6, lat=11.0)` (same **lat** as tiles 3 and 4).

**Projected coordinates**
- `x(q) = 0.6 · 0.981627 = 0.588976`, `y(q)=11`.
- Tile 3: `(x=0, y=11)` → Δx=0.588976, Δy=0 → **d=0.588976**.
- Tile 4: `(x=0.981627, y=11)` → Δx=|0.981627−0.588976|=0.392651, Δy=0 → **d=0.392651**.

**Chosen tile**: **tile_id = 4** (closest longitude when latitude is equal).

---

## 3) `test_diagonal_cosine_metric_east_wins_over_raw_degree_north`
**Purpose**: show why scaling **lon by cos(lat₀)** matters at high latitude.

**Grid (2×2) with only two sea cells**
- `lons = [0.5, 1.0]` (east has the **larger** longitude 1.0).
- `lats = [60.0, 60.45]` (north has the **larger** latitude 60.45).

**Sea mask (True = sea)**
```
# indices: j = row = latitude index; i = column = longitude index
# j↓ (south→north), i→ (west→east)

j\i   0 (lon=0.5)   1 (lon=1.0)
 1 (lat=60.45)   T (NORTH)     F
 0 (lat=60.00)   F            T (EAST)
```
**tile_id assignment over True cells (row‑major)**
- tile 0 = **EAST**  = (lon=1.0,  lat=60.0) at (j=0,i=1)
- tile 1 = **NORTH** = (lon=0.5,  lat=60.45) at (j=1,i=0)

**Reference latitude**
- `lat₀ = mean([60.0, 60.45]) = 60.225°` → `cos(lat₀) ≈ 0.496596`.

**Query**
- `q = (lon=0.5, lat=60.0)` (the south‑west corner).

### Intuition in raw degrees (WRONG metric)
- To EAST: Δlon = 0.5, Δlat = 0.0 → “0.5”.
- To NORTH: Δlon = 0.0, Δlat = 0.45 → “0.45”.
- **Raw degrees** would (incorrectly) prefer **NORTH** (0.45 < 0.5).

### Correct projected metric (x = lon · cos(lat₀), y = lat)
- `x(q) = 0.5 · 0.496596 = 0.248298`, `y(q)=60.0`.
- EAST:  `x=1.0 · 0.496596 = 0.496596`, `y=60.0`  → Δx=0.248298, Δy=0 → **d=0.248298**.
- NORTH: `x=0.5 · 0.496596 = 0.248298`, `y=60.45` → Δx=0, Δy=0.45 → **d=0.45**.

**Chosen tile**: **tile_id = 0 (EAST)** because **0.248298 < 0.45**.

**Clarification on “east”**
- We use the standard convention: longitude **increases eastward**.
- In this grid: `lon=1.0` is east of `lon=0.5`. The “EAST” sea cell is therefore the one with **lon=1.0**.

---

## Takeaways
- The projection `x = lon · cos(lat₀)` makes longitude differences commensurate with latitude differences, so Euclidean distance in (x,y) approximates true horizontal distance near `lat₀`.
- Scenarios (1) and (2) confirm the obvious along-axis behavior; scenario (3) shows why the cosine scaling is essential at higher latitudes and near diagonals.

