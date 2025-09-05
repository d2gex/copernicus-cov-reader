# LAT\_BANDING\_THEORY.md

## Purpose

When mapping haul points to nearest sea‐tile centers, we scale longitude by `cos(lat₀)` so Euclidean distance in $(x, y) = (lon·cos(lat₀), lat)$ approximates local metric distances.
For a **large** north–south domain, a single `lat₀` causes noticeable error because `cos(φ)` varies with latitude.
**Fix:** split the bounding box into **latitude bands** and use a band-specific `lat₀` per band.

---

## Small-cell area model (context)

For a small rectangular cell of size `Δ° × Δ°` at latitude `φ`:

* E–W width ≈ `111.32 km/° · Δ · cos φ`
* N–S height ≈ `111.32 km/° · Δ`
* **Area**

  $$
  A(\phi)\;\approx\;K\,\Delta^2\,\cos\phi,\qquad K = 111.32^2~\text{km}^2/\text{deg}^2.
  $$

Because nearest-neighbor behavior is sensitive to relative horizontal/vertical scaling, controlling the variation of `cos φ` inside each band keeps the KD metric consistent.

---

## Why latitude bands help

Inside a latitude band centered at `φ₀`, we approximate `cos φ` by `cos φ₀`.
Bounding the **change of `cos φ`** inside the band bounds the **area (and metric) error**.

---

## Error bound via Mean Value Theorem (MVT)

For **radian** angles,

$$
\frac{d}{d\phi}\cos\phi = -\sin\phi.
$$

If the band half-height is `h` (in **radians**), then for any `φ` with `|φ−φ₀| ≤ h`:

$$
|\cos\phi - \cos\phi_0|
\;\le\;
\max_{\phi\ \text{in band}} |\sin\phi|\cdot |\,\phi-\phi_0\,|
\;\le\;
h\cdot \sin(\phi_{\text{worst}}),
$$

where `φ_worst` is the **northern** edge in your domain (safest upper bound for `sin φ`).

**Convert back to area error per Δ×Δ cell:**

$$
|A(\phi)-A(\phi_0)|\;\le\;K\,\Delta^2\;\big|\cos\phi - \cos\phi_0\big|
\;\le\;K\,\Delta^2\;h\;\sin(\phi_{\text{worst}}).
$$

> **Units note:** The derivative formula above is valid **only when φ is in radians**.
> If you prefer degrees, convert:
> $\displaystyle \frac{d}{d(\text{deg})}\cos\phi = -\sin\phi\cdot \frac{\pi}{180}$.

---

## Band sizing rule

Given a **max per-cell area error** $E$ (km²), cell size `Δ` (degrees), and worst latitude `φ_worst` (radians):

1. **Solve for band half-height** $h$ (radians):

$$
h_{\text{rad}} \;\le\; \frac{E}{K\,\Delta^2\,\sin(\phi_{\text{worst}})}.
$$

2. **Band height** (degrees) and **number of bands** for a latitude span `S_deg`:

$$
h_{\text{deg}} = h_{\text{rad}}\cdot \frac{180}{\pi},\qquad
\text{band\_height}_{\deg} = 2\,h_{\deg},\qquad
n \;\ge\; \left\lceil \frac{S_{\deg}}{2\,h_{\deg}} \right\rceil.
$$

Why the **`2h`**? The band extends `h` below and `h` above its center → full height `2h`.

---

## Worked example (your bbox, Δ = 0.083°)

* BBox lat span $S_{\deg} = 43.514585 - 14.494878 = 29.019707^\circ$
* $K = 111.32^2 \approx 12392.14$
* $\Delta^2 = 0.083^2 = 0.006889$
* $\phi_{\text{worst}} = 43.514585^\circ \Rightarrow \sin \phi_{\text{worst}} \approx 0.689$

**Target $E = 3$ km²**:

* $h_{\text{rad}} = \dfrac{3}{12392.14 \cdot 0.006889 \cdot 0.689} \approx 0.05099$ rad
* $h_{\deg} \approx 2.921^\circ$ → band height $\approx 5.842^\circ$
* $n = \left\lceil 29.0197 / 5.842 \right\rceil = \mathbf{5}$ bands

**Target $E = 5$ km²**:

* Scale $h$ by $5/3$: $h_{\deg} \approx 4.868^\circ$ → band height $\approx 9.737^\circ$
* $n = \left\lceil 29.0197 / 9.737 \right\rceil = \mathbf{3}$ bands

**Rule of thumb for your domain:**

* ≤ **3 km²** per-cell area error → **5 latitude bands**
* ≤ **5 km²** per-cell area error → **3 latitude bands**

---

## Practical checklist

* Use **latitude bands**; within each, set `lat₀ = mean(lat)` of points (or band mid).
* Build the KD metric per band: $x = \text{lon}\cdot\cos(\text{lat}_0)$, $y=\text{lat}$.
* Keep cell size `Δ` consistent with the product you sample (e.g., 0.083°, 0.05°, 0.04°).
* If you tighten `E` or shrink `Δ`, the required number of bands **decreases**.

---

## Quick reference (formula snippet)

```text
Given:
  E_max [km^2], Δ [deg], φ_worst [deg], span S_deg [deg].

Compute:
  K = 111.32^2
  d2 = Δ^2
  s  = sin(φ_worst * π/180)
  h_rad = E_max / (K * d2 * s)
  h_deg = h_rad * 180/π
  bands = ceil(S_deg / (2 * h_deg))
```

---

## Limitations

* Small-cell planar approximation (good for typical ocean grid spacings).
* Bound is conservative (uses `sin(φ_worst)`); actual error is often lower.
* Controls **per-cell area** error; it’s a practical proxy for the longitudinal scaling error that affects nearest-neighbor behavior.
