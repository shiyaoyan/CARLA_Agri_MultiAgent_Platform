# Yantai satellite road reference

`YantaiSatelliteRebuilt.xodr` is the exported road reference from the
2026-10-13 historical Yantai satellite-scene regeneration. It is used for
offline mission previews and reproducibility, not as proof of a loaded CARLA
map or a completed drive. Import the corresponding FBX/XODR/texture bundle
into CARLA separately. The full-process runner reads the live map topology.

Coordinate source: historical 1532 x 2040 image, 0.09 m/pixel, anchor
longitude 121.51276190 / latitude 37.43327171 at pixel (520, 1319.1).
CARLA XY: ((px - 520) * .09, (py - 1319.1) * .09).

The new two-pass harvesting mission is designed geometry on this road map,
not a measured harvesting trajectory. Historical full-process measurements
from the former synthetic scenario are not new results for this map.
