# Photo Effect - Live Video Effects using OpenCV and MediaPipe

A real-time computer vision application that applies various photo effects and filters to live video feed using hand gestures for control. Built with OpenCV and MediaPipe for seamless interaction and professional visual effects.

## 🎯 Features

### Real-Time Filters

- **Black & White**: Classic monochrome conversion
- **Invert**: Color inversion effect  
- **Thermal**: Heat-map style visualization
- **Depth**: Depth-map effect with bone colormap

### Face-Based Effects

- **Dog Ears**: Animated dog ears overlay on detected faces
- **Goofy Eyes**: Large cartoon eyes replacement
- **Pixelate Face**: Real-time face pixelation for privacy
- **Moustache**: Virtual moustache overlay

### Gesture Control

- **Hand Tracking**: Real-time hand detection and landmark tracking
- **Pinch Gestures**:
  - Right hand pinch: Next effect
  - Left hand pinch: Previous effect
- **Region of Interest (ROI)**: Use both hands to create custom filter regions
- **Debounce Control**: Prevents accidental rapid switching

## 🚀 Demo

The application uses your webcam to:

1. Detect faces and hands in real-time
2. Apply selected effects automatically
3. Switch between effects using simple pinch gestures
4. Create custom filter regions using hand positioning

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- Webcam/Camera device
- Operating System: Windows, macOS, or Linux

### Setup Methods

#### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/shadow30812/photo_effect.git
cd photo_effect

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

#### Option 2: Using Conda Environment

```bash
# Clone the repository
git clone https://github.com/shadow30812/photo_effect.git
cd photo_effect

# Create and activate conda environment
conda env create -f environment.yml
conda activate photo_editor

# Run the application
python main.py
```

### Dependencies

- `opencv-python`: Computer vision and image processing
- `mediapipe`: Machine learning solutions for pose/face/hand detection
- `numpy`: Numerical computing and array operations

## 🎮 Usage

### Starting the Application

```bash
python main.py
```

### Controls

- **Q key**: Quit the application
- **Right Hand Pinch**: Switch to next effect
- **Left Hand Pinch**: Switch to previous effect
- **Two Hands**: Create region-of-interest for selective filtering
- **Close Window**: Alternative exit method

### Hand Gestures

1. **Pinch Gesture**: Bring thumb and index finger close together (within 30 pixels)
2. **ROI Selection**: Use both hands' index fingers to define rectangular region
3. **Face Detection**: Effects automatically apply when face is detected

## 🏗️ Project Structure

```
photo_effect/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── environment.yml      # Conda environment specification
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

## 🔧 Technical Details

### Architecture

- **MediaPipe Hands**: Hand landmark detection with 21 key points
- **MediaPipe FaceMesh**: Face landmark detection with 468 points
- **OpenCV**: Video capture, image processing, and display
- **Real-time Processing**: Optimized for 30+ FPS performance

### Key Parameters

- `min_detection_confidence`: 0.7 (hands), 0.5 (face)
- `min_tracking_confidence`: 0.5
- `PINCH_THRESHOLD`: 30 pixels
- `PINCH_DEBOUNCE_TIME`: 0.5 seconds

### Effect Categories

- **ROI Filters**: Applied to custom regions defined by hand positions
- **Face Effects**: Automatically applied to detected faces
- **Global Effects**: Applied to entire frame

## 🎨 Customization

### Adding New Effects

1. Create a new effect function following the existing pattern:

```python
def your_custom_effect(frame, **kwargs):
    # Your effect implementation
    return processed_frame
```

2. Add to the appropriate category list:

```python
# For ROI filters
ROI_FILTERS.append(your_custom_effect)

# For face effects  
FACE_EFFECTS.append(your_custom_effect)
```

3. Update the effect names list for display

### Modifying Gesture Controls

- Adjust `PINCH_THRESHOLD` for gesture sensitivity
- Modify `PINCH_DEBOUNCE_TIME` for switching speed
- Customize hand landmark indices for different gestures

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**: Ensure webcam is connected and not used by other apps
2. **Poor hand detection**: Ensure good lighting and clear hand visibility
3. **Slow performance**: Close other applications, reduce video resolution
4. **Import errors**: Verify all dependencies are installed correctly

### Performance Optimization

- Use adequate lighting for better detection
- Keep hands within camera view
- Ensure stable camera positioning
- Close unnecessary background applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-effect`)
3. Commit your changes (`git commit -m 'Add amazing effect'`)
4. Push to the branch (`git push origin feature/amazing-effect`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex image processing operations
- Test new effects across different lighting conditions
- Ensure backward compatibility

## 📄 License

This project is open source. Feel free to use, modify, and distribute according to your needs.

## 🙏 Acknowledgments

- **MediaPipe Team**: For providing excellent ML solutions
- **OpenCV Community**: For comprehensive computer vision tools
- **Computer Vision Community**: For inspiration and techniques

## 📧 Contact

- GitHub: [@shadow30812](https://github.com/shadow30812)
- Repository: [photo_effect](https://github.com/shadow30812/photo_effect)

---

**Built with ❤️ using Python, OpenCV, and MediaPipe**
