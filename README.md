# EasilyGetTargetFaceWithVideo

This Python script enables you to extract photos of a target face from videos featuring the target individual.

## Setup

1. Place the images of your target face in the **TargetImgToSearch** folder.
2. Store additional images for testing in the **TargetImgToTest** folder.
3. Add your target video(s) to the **inputVideos** directory.

## How to Use

1. Run the **run.cmd** file.
2. Your output will be generated and saved in the **outputImages** folder.

## Important Notes

- **TargetImgToSearch**: This folder should contain images of the target face you want to identify in the videos.
- **TargetImgToTest**: Additional images for testing purposes should be placed here.
- **inputVideos**: Store your target videos in this directory.
- **outputImages**: The output images will be saved in various subfolders within this directory.

## Dependencies

Ensure you have the following dependencies installed:

- **OpenCV**: A library for computer vision and machine learning.
- **face_recognition**: A Python library for face recognition tasks.
- **numpy**: A library for numerical computing.

## How It Works

1. The script processes the videos in the **inputVideos** folder.
2. It analyzes each frame of the video to detect faces.
3. If a detected face matches the target face, a photo is captured and saved in the **outputImages/target** folder.
4. If a detected face doesn't match the target face, it's categorized as noise and saved in the **outputImages/noise** folder.
5. Faces that couldn't be identified are saved in the **outputImages/otherFaces** folder.
6. Frames without any detected faces are saved in the **outputImages/imgsWithoutAnyFace** folder.

## Tips for Better Results

- Ensure that your target face images are clear and well-lit.
- Use multiple target images for better accuracy.
- Experiment with different videos to improve face detection in various scenarios.
