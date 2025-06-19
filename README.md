# Skin Lesion Classifier

A comprehensive medical analysis tool for skin lesion assessment using advanced computer vision and medical criteria.

## Features

- **Image Upload & Analysis**: Upload skin lesion images for detailed analysis
- **ABCDE Criteria**: Evaluates asymmetry, border irregularity, color variation, diameter, and evolution
- **Skin Type Integration**: Fitzpatrick skin type classification for improved accuracy
- **Risk Assessment**: Provides confidence scores and risk categorization
- **Medical Guidelines**: Follows established dermatological assessment protocols

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV, PIL
- **Machine Learning**: PyTorch, scikit-learn
- **Medical Analysis**: Custom ABCDE algorithm implementation

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rudravaish/Flask.git
   cd Flask
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

5. **Access the application**:
   Open your browser and go to `http://127.0.0.1:5001`

## Usage

1. **Upload Image**: Select a clear image of the skin lesion
2. **Select Skin Type**: Choose your Fitzpatrick skin type for improved accuracy
3. **Choose Body Location**: Select the body part where the lesion is located
4. **Get Analysis**: Review the detailed ABCDE analysis and recommendations

## Medical Disclaimer

This tool is designed for educational and screening purposes only. It should not replace professional medical evaluation, diagnosis, or treatment. Always consult with a qualified dermatologist for definitive medical advice.

## Contributing

This project was developed as a medical analysis tool. Contributions are welcome, but please ensure all medical-related changes follow established clinical guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by Rudra Vaishnav as a comprehensive skin lesion analysis tool. 