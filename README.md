# Realtime Video Analysis

A comprehensive real-time video analysis system that can perform multiple computer vision tasks including human detection, object detection, emotion recognition, and vehicle detection.

## Features

- Human Detection
- Object Detection
- Emotion Recognition
- Vehicle Detection
- Motion Detection
- Real-time Video Processing
- Web-based Interface

## Prerequisites

- Python 3.7+
- Webcam or Video Input Device
- Modern Web Browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/realtime-video-analysis.git
cd realtime-video-analysis
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download YOLOv3 weights:
   - Download the YOLOv3 weights file from [here](https://pjreddie.com/media/files/yolov3.weights)
   - Place the downloaded file in the `python_Scripts` directory

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Select the desired analysis mode:
   - Human Detection
   - Object Detection
   - Emotion Recognition
   - Vehicle Detection
   - Motion Detection

## Project Structure

```
realtime/
├── app.py                 # Main Flask application
├── python_Scripts/        # Core analysis scripts
│   ├── human_yolov3.py   # Human detection module
│   ├── object_yolov3.py  # Object detection module
│   └── traffic_yolov3.py # Vehicle detection module
├── static/               # Static files (CSS, JS, images)
└── templates/            # HTML templates
```

## Technologies Used

- Flask: Web framework
- OpenCV: Computer vision tasks
- TensorFlow: Deep learning framework
- YOLOv3: Object detection
- FER: Facial emotion recognition
- NumPy: Numerical computations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- YOLOv3 for object detection
- FER for emotion recognition
- OpenCV community
