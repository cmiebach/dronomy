"""dronomy_loc — GPS-denied visual localization of drone footage against
georeferenced satellite imagery (IE x Dronomy capstone).

Pipeline overview
-----------------
    video frame ──► [matching] ──► drone↔reference homography
                                        │
    reference tile (georeferenced) ─────┘
                                        ▼
                         image-center pixel ──► [geo] ──► (lat, lon, yaw)

Sub-packages
------------
    data       frame extraction from the drone video (OpenCV)
    reference  pluggable satellite-imagery providers (IGN, Google Earth Engine)
    matching   pluggable matchers (classical SIFT/ORB, deep LoFTR/SuperGlue)
    localize   ties matching + georeferencing into a per-frame pose estimate
    viz        overlays and trajectory plots
"""

__version__ = "0.1.0"
