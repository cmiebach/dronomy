# Project Guide

Complete walkthrough of the Dronomy GPS denied localization project. Written so anyone can open it, understand the goal, and find any piece of code fast.

This document has three layers:

1. Technical reference, folder by folder and function by function
2. Plain English version of the same thing
3. Future steps, questions for the professor, and open unknowns

***

## 1. What the project does (one screen)

* Goal: figure out where a drone was (latitude, longitude, and heading) using only the video from its downward facing camera, with no GPS
* Method: take a frame from the drone video, find the same spot on a satellite image of the area, and convert that match into real world coordinates
* The satellite image is georeferenced, meaning every pixel has a known latitude and longitude, so once a frame is matched we can read off the position
* The whole flow:
  * drone frame goes into a matcher
  * satellite tile (with known coordinates) goes into the same matcher
  * matcher returns a homography (a 2D mapping from frame pixels to satellite pixels)
  * we send the frame center through that mapping, land on a satellite pixel, and read its latitude and longitude

***

## 2. Folder by folder map

* `config/`
  * `config.yaml`: every setting in one place (video path, rough location, which satellite provider, which matcher, RANSAC thresholds, output paths)
* `src/dronomy_loc/`: the actual Python package, split by job
  * `data/`: reading the video and pulling out frames
  * `reference/`: getting the satellite image and all coordinate math
  * `matching/`: finding correspondences between a frame and the satellite image
  * `localize/`: turning a match into a latitude, longitude, and heading
  * `viz/`: drawing overlays and trajectory plots
* `scripts/`: small runnable programs you call from the command line (the demo pieces)
* `tests/`: automatic checks for the coordinate math
* `docs/`: literature review and report outline
* `data/`: everything the code generates (frames, satellite tiles, outputs). Ignored by git so the repo stays small
* `project_instructions/`: the brief (slide deck) and the meeting transcript
* `dronomy_video/`: the raw 4K drone video (ignored by git, far too large to commit)

***

## 3. File and function reference (technical)

### `config/config.yaml`
* Plain text settings file loaded at startup
* Holds: video path and the rough latitude and longitude parsed from the filename, frame sampling rate, satellite provider choice, tile size in meters and pixels, matcher choice and its parameters, RANSAC thresholds, output locations

### `src/dronomy_loc/__init__.py`
* Package marker and top level docstring describing the pipeline
* Sets `__version__`

### `src/dronomy_loc/config.py`
* `REPO_ROOT`: absolute path to the repo root, computed from this file location
* `DEFAULT_CONFIG_PATH`: path to `config/config.yaml`
* `_to_namespace(obj)`: recursively turns a dictionary into a dot access object, so you can write `cfg.frames.out_dir` instead of `cfg["frames"]["out_dir"]`
* `load_config(path)`: reads the YAML file, converts it to the dot access object, attaches `repo_root`, returns it
* `resolve(path_str)`: turns a path written in the config (relative to the repo) into a full absolute path

### `src/dronomy_loc/data/frames.py`
* `FrameInfo`: a small record holding a frame index, its timestamp in seconds, and the image array
* `_resize_long_edge(img, long_edge)`: shrinks an image so its longest side equals `long_edge`, keeping aspect ratio, used to speed up matching on 4K frames
* `iter_frames(video_path, every_n_seconds, max_frames, resize_long_edge)`: a generator that opens the video and yields one `FrameInfo` every N seconds. Uses `grab` then `retrieve` so it skips decoding frames it does not need, which is fast and robust on Windows
* `extract_frames(...)`: same sampling, but writes each frame to disk as a JPEG named with its index and timestamp, returns the list of file paths
* `probe(video_path)`: returns basic video facts (frames per second, frame count, width, height, duration) without any external tool like ffprobe

### `src/dronomy_loc/reference/geo.py` (the coordinate engine)
* `R_EARTH`: Earth radius constant used by the Web Mercator formulas
* `lonlat_to_mercator(lon, lat)`: converts longitude and latitude in degrees to Web Mercator meters (EPSG:3857)
* `mercator_to_lonlat(x, y)`: the inverse, meters back to degrees
* `meters_per_degree_lat(lat)`: how many meters one degree of latitude and one degree of longitude span at a given latitude, used for reporting errors in meters
* `haversine_m(lat1, lon1, lat2, lon2)`: true ground distance in meters between two coordinate pairs
* `mercator_bbox_around(lon, lat, span_m)`: builds a square bounding box in Mercator meters centered on a point, used to request a satellite tile of a chosen size
* `GeoImage`: a satellite raster bundled with its bounding box, which is what makes pixel to coordinate conversion possible
  * `width`, `height`: pixel dimensions
  * `meters_per_pixel`: ground resolution in each axis
  * `pixel_to_mercator(px, py)`: pixel position to Mercator meters
  * `mercator_to_pixel(x, y)`: Mercator meters to pixel position
  * `pixel_to_lonlat(px, py)`: pixel position to longitude and latitude
  * `lonlat_to_pixel(lon, lat)`: longitude and latitude to pixel position

### `src/dronomy_loc/reference/base.py`
* `ReferenceProvider`: the interface every satellite source must follow, with one method `fetch(lat, lon, span_meters, pixels)` returning a `GeoImage`
* `get_provider(name, cfg)`: factory that returns the right provider object for `ign` or `gee`

### `src/dronomy_loc/reference/ign.py`
* `IGNProvider`: gets French national orthophotos from the IGN Geoplateforme service, no API key needed
  * `__init__`: reads the service URL and layer name from config
  * `fetch(...)`: builds a WMS GetMap request with an explicit Mercator bounding box, downloads the image, wraps it in a `GeoImage` so coordinates are exact. Raises a clear error if the service returns an XML error instead of an image

### `src/dronomy_loc/reference/gee.py`
* `GEEProvider`: placeholder for Google Earth Engine, the source named in the brief
  * `fetch(...)`: currently raises a clear NotImplementedError with a sketch of the steps needed once authentication is set up

### `src/dronomy_loc/reference/store.py`
* `save_reference(geo, out_dir, name)`: saves a fetched tile as a PNG plus a small file holding its bounding box, so it can be reused without downloading again
* `load_reference(out_dir, name)`: loads that PNG and bounding box back into a `GeoImage`

### `src/dronomy_loc/matching/base.py`
* `MatchResult`: the standard result every matcher returns, holding matched points in the frame, matched points in the satellite tile, the homography, the inlier mask, and the raw match count
  * `n_inliers`: count of matches RANSAC kept
  * `ok`: whether a homography was found
* `estimate_homography(src_pts, dst_pts, ...)`: runs RANSAC to find the 2D mapping from frame to satellite and to reject bad matches, returns the mapping and which points were kept
* `Matcher`: the interface every matcher follows, with one method `match(drone_bgr, ref_rgb)`
* `get_matcher(method, cfg)`: factory returning the classical or deep matcher

### `src/dronomy_loc/matching/classical.py`
* `ClassicalMatcher`: the baseline, using SIFT, ORB, or AKAZE features
  * `__init__`: reads detector choice and thresholds from config
  * `_build_detector`: creates the chosen OpenCV detector
  * `_gray`: converts an image to grayscale
  * `match(...)`: detects features in both images, matches them with a ratio test to keep only confident pairs, then estimates the homography

### `src/dronomy_loc/matching/deep.py`
* `DeepMatcher`: the modern approach using LoFTR from the kornia library, strong on low texture areas like grass
  * `__init__`: reads model name, weights, and device from config, caps input size for speed
  * `_lazy_init`: imports PyTorch and kornia only when first used, with a clear install message if they are missing
  * `_prep`: turns an image into the grayscale tensor LoFTR expects, downscaling if needed
  * `match(...)`: runs LoFTR to get dense correspondences, rescales points back to original size, then estimates the homography

### `src/dronomy_loc/localize/pipeline.py`
* `PoseEstimate`: the per frame answer, holding success flag, latitude, longitude, heading in degrees, ground meters per pixel (an altitude proxy), inlier and match counts, frame index, and timestamp
* `_apply_H(H, x, y)`: pushes a single point through a homography
* `pose_from_homography(H, frame_shape, ref)`: the core conversion
  * sends the frame center through the homography to a satellite pixel, then to latitude and longitude
  * measures heading by sending the frame up direction through the homography and comparing to north on the tile
  * measures scale by seeing how far one frame pixel stretches on the tile, times the tile ground resolution
* `localize_frame(frame_bgr, ref, matcher)`: runs the matcher on one frame, and if it succeeds returns the pose plus the raw match result

### `src/dronomy_loc/viz/overlay.py`
* `draw_matches(drone_bgr, ref_rgb, mr, max_draw)`: draws the frame and tile side by side with lines connecting matched points
* `draw_frame_footprint(ref_rgb, H, frame_shape)`: draws the outline of where the frame lands on the satellite tile, plus a dot at the center (the estimated position)
* `plot_trajectory(ref, lats, lons, out_path, title)`: plots the full estimated path on top of the satellite tile and saves a PNG

### `scripts/`
* `_bootstrap.py`: lets the scripts import the package without installing it, by adding `src` to the path
* `01_extract_frames.py`: reads the video and writes sampled frames, or just prints metadata with `--probe`
* `02_fetch_reference.py`: downloads a satellite tile for the area, saves it, and prints a sanity check that the center pixel maps back to the requested coordinate
* `03_localize_frame.py`: the minimum viable demo, takes one frame, localizes it, prints latitude, longitude, and heading, and saves the match and footprint overlays
* `04_run_video.py`: runs localization across the whole video, writes a trajectory CSV and a path on map plot

### `tests/test_geo.py`
* Four checks that need no network and no PyTorch
  * Mercator conversion round trips back to the same coordinate
  * The center pixel of a tile maps back to the requested point within one meter
  * Any pixel converts to coordinates and back to the same pixel
  * The top left pixel is north and west of the bottom right pixel (orientation is correct)

***

## 4. Plain English version (no jargon)

* The drone filmed the ground looking straight down. We want to know where it was, but we are pretending we have no GPS
* A satellite photo of the same area acts like a map where we already know the coordinates of every point
* For each moment in the video we take one picture from the drone and ask: where on the satellite map does this picture sit
* The computer finds matching landmarks between the two pictures (a road edge, a curb, a manhole, a corner of a field)
* From those matches it works out the exact spot, the direction the drone was pointing, and roughly how high it was
* We then draw that spot on the map, and by doing this for the whole video we draw the drone path
* Why this is set up the way it is:
  * The satellite source can be swapped, because the brief prefers Google Earth but the professor said open data is fine, and the French national photos are free and already carry coordinates
  * The matching method can be swapped, because the brief asks us to compare a classic method against a modern AI method, so we built both behind the same switch
  * We process each frame on its own for now, because the brief said to start simple and add time based smoothing later
* What each folder is for, in one line each:
  * config: the settings dial
  * data: turns the video into pictures
  * reference: gets the map and does all the coordinate math
  * matching: finds the same landmarks in two pictures
  * localize: turns a match into a real coordinate
  * viz: makes the pictures and the path plot
  * scripts: the buttons you press to run each step
  * tests: proof that the coordinate math is correct

***

## 5. How to run it

* Inspect the video: `python scripts/01_extract_frames.py --probe`
* Extract some frames: `python scripts/01_extract_frames.py --every 2.0 --max 30`
* Get a satellite tile: `python scripts/02_fetch_reference.py --provider ign --span 1500 --pixels 4096`
* Localize one frame: `python scripts/03_localize_frame.py --frame data/frames/<one>.jpg --method classical`
* Run the whole video: `python scripts/04_run_video.py --every 2.0 --method classical`
* Run the math tests: `pytest`

***

## 6. Current status

* Done: full project skeleton, settings, coordinate engine, frame extraction, classical matcher, deep matcher, pose math, visualizations, four runnable scripts, four passing tests
* Verified: package imports cleanly, tests pass, the video metadata reads correctly (4K, about 30 frames per second, about 229 seconds, 6853 frames)
* Not yet verified on real data: we have not yet downloaded a live satellite tile or run a real match, those are the very next steps
* Not installed yet: PyTorch and kornia, needed only for the deep matcher

***

## 7. Future steps (in order)

* Download one real IGN tile and confirm the center maps back to the filename coordinate
* Localize one real frame with the classical matcher and look at the overlay
* Install PyTorch and kornia, then localize the same frame with LoFTR
* Compare the two matchers on a set of frames: success rate, number of good matches, runtime
* Run the full video and plot the path on the map
* Add a simple smoothing step so the path is steady, since the brief calls this an extension
* Bonus track: estimate motion between consecutive frames (visual odometry) and blend it with the map matches for a smoother and more reliable path
* Write the report and presentation using the overlays and the path plot

***

## 8. Questions to ask the professor (to unblock the next steps)

* Is there a flight log or telemetry file for this video, such as an SRT file with GPS and altitude, that we can use as ground truth to measure our accuracy
* What accuracy is expected, for example position error in meters, and is heading accuracy graded too
* Must we use Google Earth specifically, or is the French national open imagery acceptable as the reference, since it is already georeferenced
* Do we know the camera details, such as field of view or focal length and any lens distortion, which would let us convert pixels to meters more precisely
* Was the camera always pointing straight down, or did it tilt, since tilt changes the matching model
* Do we know the flight altitude, even approximately, since it sets the image scale
* Is real time speed graded, or is accuracy the only thing that matters
* If this video proves too hard to match, can we get the easier video that was mentioned
* For the final deliverable, what format do you want for the report and the presentation, and what are the dates

***

## 9. Open technical unknowns (things we will resolve as we go)

* How well classical features survive on grass heavy scenes, which is the main risk
* Whether the chosen tile size of 1500 meters is the right search area, too small risks missing the location, too large slows matching and adds ambiguity
* How recent and how high resolution the satellite imagery is versus the drone footage, since season and lighting differences hurt matching
* Whether we will get ground truth at all, which decides if we can report numbers or only show overlays
