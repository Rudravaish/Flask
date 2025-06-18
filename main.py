from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from model import predict_lesion
from multi_input_model import get_multi_input_prediction, get_body_part_options
import os
import logging
import json
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-change-in-production")

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cleanup_old_uploads():
    """Clean up uploaded files older than 1 hour to prevent storage issues."""
    try:
        import time
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                # Remove files older than 1 hour (3600 seconds)
                if file_age > 3600:
                    os.remove(filepath)
                    app.logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        app.logger.warning(f"Error during cleanup: {str(e)}")

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define Fitzpatrick skin types with detailed descriptions
FITZPATRICK_TYPES = {
    "I": "Very fair - Always burns, never tans (Celtic, Scandinavian)",
    "II": "Fair - Usually burns, tans poorly (Northern European)", 
    "III": "Medium - Sometimes mild burn, tans uniformly (Mixed European)",
    "IV": "Olive - Burns minimally, always tans well (Mediterranean, Asian)",
    "V": "Brown - Rarely burns, tans profusely (Middle Eastern, Latino)",
    "VI": "Dark brown/Black - Never burns, deeply pigmented (African, Aboriginal)"
}

# Body types with melanoma risk factors
BODY_TYPES = {
    "average": "Average risk profile",
    "fair_many_moles": "Fair skin with many moles (50+ moles)",
    "family_history": "Family history of melanoma",
    "previous_skin_cancer": "Previous skin cancer diagnosis",
    "immunocompromised": "Immunocompromised or taking immunosuppressive drugs",
    "frequent_sun_exposure": "Frequent sun exposure or history of severe sunburns"
}

# Skin types with potential model bias
BIAS_WARNING_TYPES = ["V", "VI"]


@app.route('/', methods=['GET', 'POST'])
def home():
    """Main route for file upload and prediction."""
    # Define Fitzpatrick skin types for template
    fitzpatrick_types = {
        'I': 'Always burns, never tans (Very fair skin)',
        'II': 'Usually burns, tans minimally (Fair skin)',
        'III': 'Sometimes burns, tans gradually (Medium skin)',
        'IV': 'Minimal burning, tans well (Olive skin)',
        'V': 'Rarely burns, tans easily (Brown skin)',
        'VI': 'Never burns, deeply pigmented (Dark brown/black skin)'
    }
    
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'image' not in request.files:
                flash('No file selected. Please choose an image to upload.', 'error')
                return redirect(request.url)
            
            file = request.files['image']
            
            # Get form parameters
            skin_type = request.form.get('skin_type', 'III')  # Default to Type III
            body_part = request.form.get('body_part', 'other')  # Default to other location
            
            # Evolution tracking (E in ABCDE)
            has_evolved = request.form.get('has_evolved') == 'yes'
            evolution_weeks = int(request.form.get('evolution_weeks', 0)) if request.form.get('evolution_weeks') else 0
            
            # Manual measurements (optional)
            manual_length = float(request.form.get('manual_length', 0)) if request.form.get('manual_length') else None
            manual_width = float(request.form.get('manual_width', 0)) if request.form.get('manual_width') else None
            
            # Check if file was actually selected
            if not file.filename or file.filename == '':
                flash('No file selected. Please choose an image to upload.', 'error')
                return redirect(request.url)
            
            # Validate file type
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, WebP).', 'error')
                return redirect(request.url)
            
            if file and file.filename and allowed_file(file.filename):
                # Secure the filename - we know filename is not None at this point
                filename = secure_filename(str(file.filename))
                
                # Create unique filename to avoid conflicts
                import time
                timestamp = str(int(time.time()))
                name, ext = os.path.splitext(filename)
                unique_filename = f"{name}_{timestamp}{ext}"
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                # Save the file
                file.save(filepath)
                app.logger.info(f"File saved to: {filepath}")
                
                # Make prediction with enhanced parameters
                try:
                    prediction_result = predict_lesion(
                        filepath, 
                        skin_type=skin_type,
                        body_part=body_part,
                        has_evolved=has_evolved,
                        evolution_weeks=evolution_weeks,
                        manual_length=manual_length,
                        manual_width=manual_width
                    )
                    
                    # Handle different return formats safely
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 3:
                        prediction, confidence, analysis_data = prediction_result
                    elif isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                        prediction, confidence = prediction_result
                        analysis_data = None
                    else:
                        # Fallback for unexpected formats
                        prediction = "Error in Analysis"
                        confidence = 0
                        analysis_data = None
                    
                    app.logger.info(f"Prediction: {prediction}, Confidence: {confidence}%")
                    
                    # Generate detailed analysis summary
                    analysis_summary = None
                    if analysis_data:
                        analysis_summary = {
                            'detected_skin_tone': analysis_data.get('detected_skin_tone', skin_type),
                            'features': {
                                'asymmetry': analysis_data.get('asymmetry', 0),
                                'border': analysis_data.get('border', 0),
                                'color': analysis_data.get('color', 0),
                                'diameter': analysis_data.get('diameter', 0)
                            },
                            'analysis_type': analysis_data.get('analysis_type', 'standard')
                        }
                    
                    # Check for bias warning
                    bias_warning = None
                    if skin_type in BIAS_WARNING_TYPES:
                        bias_warning = f"Model may have reduced accuracy on Fitzpatrick skin type {skin_type} ({FITZPATRICK_TYPES[skin_type]}). Please consult a dermatologist for professional evaluation."
                    
                    # Enhanced bias warning for darker skin tones with better analysis
                    detected_skin_tone = analysis_data.get('detected_skin_tone') if analysis_data else skin_type
                    if detected_skin_tone in ['V', 'VI'] and analysis_data:
                        enhanced_warning = f"Analysis optimized for darker skin tone (Type {detected_skin_tone}). This analysis accounts for melanin-specific patterns and post-inflammatory changes common in darker skin."
                    else:
                        enhanced_warning = None
                    
                    # Clean up old uploaded files to prevent storage issues
                    cleanup_old_uploads()
                    
                    return render_template('index.html', 
                                         result=prediction, 
                                         confidence=confidence, 
                                         image_path=filepath,
                                         filename=unique_filename,
                                         skin_type=skin_type,
                                         skin_type_description=FITZPATRICK_TYPES[skin_type],
                                         bias_warning=bias_warning,
                                         enhanced_warning=enhanced_warning,
                                         analysis_summary=analysis_summary,
                                         detected_skin_tone=detected_skin_tone,
                                         fitzpatrick_types=FITZPATRICK_TYPES)
                except Exception as e:
                    app.logger.error(f"Prediction error: {str(e)}")
                    flash('Error processing image. Please try again with a different image.', 'error')
                    # Clean up the uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return redirect(request.url)
                    
        except Exception as e:
            app.logger.error(f"Upload error: {str(e)}")
            flash('An error occurred while processing your upload. Please try again.', 'error')
            return redirect(request.url)
    
    return render_template('index.html', fitzpatrick_types=FITZPATRICK_TYPES, body_part_options=get_body_part_options())

@app.route('/advanced')
def advanced_analysis():
    """Advanced analysis interface with EfficientNetB0 multi-input model"""
    return render_template('advanced_analysis.html')

@app.route('/predict', methods=['POST'])
def predict_multi_input():
    """
    Advanced prediction endpoint using EfficientNetB0 multi-input model
    Accepts image and patient metadata, returns comprehensive analysis
    """
    try:
        # Validate file upload
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a valid image.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        # Extract metadata from form or JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Validate and extract required parameters
        try:
            age = float(data.get('age', 50))
            uv_exposure = float(data.get('uv_exposure', 5))
            family_history = int(data.get('family_history', 0))
            skin_type = int(data.get('skin_type', 3))
            body_part = int(data.get('body_part', 10))
            evolution_weeks = float(data.get('evolution_weeks', 0))
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid parameter format: {str(e)}'}), 400
        
        # Validate parameter ranges
        if not (0 <= age <= 120):
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if not (0 <= uv_exposure <= 10):
            return jsonify({'error': 'UV exposure must be between 0 and 10'}), 400
        if family_history not in [0, 1]:
            return jsonify({'error': 'Family history must be 0 or 1'}), 400
        if not (1 <= skin_type <= 6):
            return jsonify({'error': 'Skin type must be between 1 and 6'}), 400
        if not (0 <= body_part <= 19):
            return jsonify({'error': 'Body part must be between 0 and 19'}), 400
        if evolution_weeks < 0:
            return jsonify({'error': 'Evolution weeks must be non-negative'}), 400
        
        # Get prediction from multi-input model
        try:
            prediction_results = get_multi_input_prediction(
                filepath, age, uv_exposure, family_history,
                skin_type, body_part, evolution_weeks
            )
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            app.logger.info(f"Multi-input prediction completed: Risk={prediction_results['risk_level']}")
            
            return jsonify({
                'success': True,
                'results': prediction_results
            })
            
        except Exception as e:
            # Clean up uploaded file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            app.logger.error(f"Multi-input prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
    except Exception as e:
        app.logger.error(f"Multi-input endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Please upload an image smaller than 16MB.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    app.logger.error(f"Internal server error: {str(e)}")
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('home'))

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any uncaught exceptions."""
    app.logger.error(f"Unhandled exception: {str(e)}")
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
