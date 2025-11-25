# Face-Track
Real–time intelligent camera framing and face trackin using computer vision

## Overview
**Face-Track** is a real-time computer vision system that transforms any standard webcam into a software-driven camera that can frame and zoom in on a user's face.
The pipeline automatically detects faces, centres subjects and applies a smooth digital zoom and tracking.

The goal of the project is to provide production-level quality to remote meetings, content creation and live streaming using only classical CV techniques and efficient real time processing.

## Features
- **Real time tracking** – automatically centres and follows human subjects across the frame
- **Gesture controlled zooming** 
   - when the user makes the rock symbol (🤟) with their hand, the camera zooms in
   - when the user makes the peace symbol (✌️) with their hand, the camera zooms out
   - 

## Tech stack
- **Python3**
- **OpenCV** – used in real-time capture and processing
- **MediaPipe** – used for its landmark detection features in face tracking and gesture tracking
- **NumPy** – used in image and matrix operations
