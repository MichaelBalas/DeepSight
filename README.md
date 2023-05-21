# DeepSight


## Introduction
This repository contains the open-source code for real-time iris depth estimation using Google's MediaPipe Iris model. This innovative tool aims to provide proactive measures against the global rise of myopia (nearsightedness) by tracking and providing feedback on an individual's screen-viewing habits.

The method presented exploits the constant physical size of the human iris to estimate the distance between the observer and the screen using a standard RGB camera. Furthermore, this repository also includes a script to calculate real-world iris size based on images or videos taken at known distances, potentially enhancing the accuracy of depth estimation.

## Installation
### Prerequisites
Ensure you have the following installed on your machine:

- Python 3.7+
- pip
- Git


Clone this repository to your local machine:
```
git clone https://github.com/MichaelBalas/DeepSight.git
```

Navigate to the cloned repository:
```
cd DeepSight
```

Install the required packages:
```
pip install -r requirements.txt
```

**You will have to specify which device you're using so the focal length can be determined**.

We provice typical focal lengths (in pixels) for most phones and computers, however you may have to adjust these values for your particular device.

## Usage
### Real-Time Iris Depth Estimation
Run the script with:
```
python irisDepth.py -d computer
```
This will initiate the webcam and start estimating the iris depth in real-time.

### Pre-Recorded Video-Based Iris Depth Estimation
```
python irisDepth.py -i your_video.mp4 -d phone
```
This will load your pre-recorded video (taken by your phone) and start estimating the iris depth.

### Iris Size Estimation
To calculate your real-world iris size, first take an image or video at a pre-specified distance from your device (e.g. 30cm, or a standard ruler). Then use:
```
python irisDiameter.py -i your_img.jpg -d phone -z 30
```
or
```
python irisDiameter.py -i your_video.mp4 -d phone -z 30
```
Remember to provide the known distance when capturing the image or video. The entirety of the video should be taken at a single distance, since iris size will be computed and averaged across all video frames.

## Contributing
We welcome contributions to improve this open-source project. To contribute, please:

## Fork the repository
1. Create your feature branch (git checkout -b feature/yourFeature)
2. Commit your changes (git commit -am 'Add some feature')
3. Push to the branch (git push origin feature/yourFeature)
4. Create a new pull request

## Support
For any questions or issues, please raise an issue in the repository, or contact us directly at 1michaelbalas@gmail.com

Check out the [Google MediaPipe](https://github.com/google/mediapipe) Github along with [other demos](https://github.com/Rassibassi/mediapipeDemos) to learn more.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


**Funding:** None<br/> 
**Conflicts of Interest:** None<br/> 
*This project is not affiliated with Google or the official MediaPipe solutions.*
