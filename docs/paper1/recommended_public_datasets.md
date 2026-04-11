# Recommended Public Datasets For Paper 1

## Chosen shared label space
- `happy`
- `sad`
- `neutral`
- `angry`

This four-class setting is the most practical shared subset for the current Paper 1 repo because it maps cleanly to common public emotion datasets and avoids forcing unsupported labels into the benchmark.

## Recommended dataset roles

### RAVDESS
- best immediate public dataset to download and run now
- open-access audiovisual data with clear filename-based emotion labels
- practical first upgrade because it works well with the current replay-through-ROS path

### RAF-DB
- best primary choice for controlled face-image evaluation
- useful for the offline dataset-evaluation figures and tables
- fits the current face-based baseline well

### CREMA-D
- best practical choice for dataset replay through the ROS pipeline
- includes acted emotional audiovisual clips that can be sampled into replayable frames
- useful for bridging offline perception testing and ROS2 dataset replay

## Why this is stronger than the local pilot room images
- public datasets provide a clearer and more defensible test set
- labels are structured and repeatable
- the replay path stays controlled instead of relying on ad hoc webcam captures

## Current recommendation
- use `RAVDESS` right now as the immediate runnable public dataset
- use `RAF-DB` for the main controlled image-evaluation tables and figures once downloaded
- use `CREMA-D` for replay-through-ROS figures and multimodal discussion
- keep the live hybrid Windows-camera run as runtime integration evidence, not the main perception benchmark
