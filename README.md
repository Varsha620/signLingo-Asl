# ASL Fingerspelling Recognition System

## Overview
The **ASL Fingerspelling Recognition System** is a deep learning-based project designed to recognize American Sign Language (ASL) fingerspelling gestures using computer vision techniques. The system utilizes **MediaPipe Hands** for hand tracking, **OpenCV** for real-time video processing, and a **deep learning model** trained on ASL hand gesture landmarks.

## Features
- Real-time ASL fingerspelling recognition using webcam input.
- Uses **MediaPipe Hands** to extract 21 hand landmarks.
- A trained deep learning model classifies gestures into ASL letters (A-Z).
- Displays the predicted ASL letter on-screen.
- Accumulates recognized letters to form words and sentences.

## System Requirements
- Python 3.x
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- Webcam (for real-time gesture recognition)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/asl-fingerspelling-recognition.git
   cd asl-fingerspelling-recognition
   ```
2. **Install Required Libraries**:
   ```bash
   pip install tensorflow opencv-python mediapipe numpy
   ```
3. **Run the Application**:
   ```bash
   python recognize_asl.py
   ```

## Dataset
The system is trained on a dataset of ASL hand gestures. It includes:
- Preprocessed hand landmark data extracted using **MediaPipe Hands**.
- Data augmentation techniques to improve model robustness.
- A deep learning classifier trained using **TensorFlow and Keras**.

## Model Architecture
The deep learning model consists of:
- **Input Layer**: Processes 21 hand landmark coordinates.
- **Hidden Layers**: Fully connected layers with ReLU activation.
- **Dropout Layers**: Prevents overfitting.
- **Output Layer**: Uses softmax activation to classify letters A-Z.

## How It Works
1. Captures live video from the webcam.
2. Detects and tracks hand landmarks using **MediaPipe Hands**.
3. Extracts and normalizes hand landmark features.
4. Passes the features to the trained model for classification.
5. Displays the predicted ASL letter in real-time.

## Testing and Performance
- Achieved **96% accuracy** on the test dataset.
- Real-time recognition with low latency.
- Performs well under different lighting conditions.
- Challenges: Similar gestures (e.g., 'M' vs 'N') may be misclassified.

## Future Enhancements
- Improve accuracy for visually similar gestures.
- Add support for dynamic ASL words (not just fingerspelling).
- Develop a mobile app for easier accessibility.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Added new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

## License
This project is open-source and available under the **MIT License**.

## Contact
For any questions or suggestions, feel free to reach out or open an issue in the repository.

---
