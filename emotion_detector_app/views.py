from django.shortcuts import render
from django.conf import settings
from django.core.files.base import ContentFile
from .models import EmotionRecord
import tensorflow as tf
import numpy as np
import cv2
import os

# Load the Trained Model
model_path = os.path.join(settings.BASE_DIR, 'ENYI_EMOTION_MODEL.keras')
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the Face Detector
# Path to the cascade file (in the root project directory)
face_cascade_path = os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_default.xml')
try:
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load face cascade from {face_cascade_path}")
    print(f"Face cascade loaded successfully from {face_cascade_path}")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    face_cascade = None


# Emotion Classes and Image Dimensions
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_HEIGHT = 48
IMG_WIDTH = 48


def home(request):
    """
    This view handles both GET (display page) and POST (process image) requests.
    """
    context = {
        'prediction': None,
        'original_image_base64': None, # To display the original image
        'face_image_base64': None,     # To display the detected face
    }
    
    print(f"\nRequest Method: {request.method}")

    if request.method == 'POST':
        print("--- INSIDE POST BLOCK ---")

        if model is None:
            print("Error: Model is not loaded.")
            context['prediction'] = 'Error: Model is not loaded.'
            return render(request, 'emotion_detector/index.html', context)
        
        if face_cascade is None:
            print("Error: Face detector is not loaded.")
            context['prediction'] = 'Error: Face detector is not loaded.'
            return render(request, 'emotion_detector/index.html', context)


        uploaded_file = request.FILES.get('file')
        
        if uploaded_file:
            print(f"Found file: {uploaded_file.name}")
            try:
                in_memory_file = uploaded_file.read()
                np_arr = np.frombuffer(in_memory_file, np.uint8)
                img = cv2.imdecode(np_arr, cv2.COLOR_BGR2GRAY)

                if img is None:
                    raise ValueError("Could not decode image.")

                # Encode original image for display
                # We'll need base64 to embed it in HTML
                import base64
                _, buffer = cv2.imencode('.png', img)
                context['original_image_base64'] = base64.b64encode(buffer).decode('utf-8')

                # Detect faces in the image
                # scaleFactor: decreases image size at each step
                # minNeighbors: how many neighbors each candidate rectangle should have
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    context['prediction'] = 'No face detected.'
                    print("No face detected.")
                else:
                    # For simplicity, we'll take the first face found
                    (x, y, w, h) = faces[0]
                    
                    # Crop the face from the original image
                    face_crop = img[y:y+h, x:x+w]
                    
                    # Encode face crop for display
                    _, buffer_face = cv2.imencode('.png', face_crop)
                    context['face_image_base64'] = base64.b64encode(buffer_face).decode('utf-8')


                    # Continue preprocessing the cropped face
                    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (IMG_WIDTH, IMG_HEIGHT))
                    
                    img_array = np.expand_dims(resized_face, axis=-1)
                    img_array = np.expand_dims(img_array, axis=0)

                    prediction = model.predict(img_array)
                    predicted_index = np.argmax(prediction)
                    predicted_label = CLASS_NAMES[predicted_index]
                    confidence_score = prediction[0][predicted_index]
                    
                    print(f"Prediction complete: {predicted_label}")
                    context['prediction'] = f"The model predicts: {predicted_label} (Confidence: {confidence_score*100:.2f}%)"

                    # Use username from form
                    form_username = request.POST.get('username', 'Anonymous')
                    if request.method == 'POST':
                        if not form_username: # Handle empty string
                            form_username = 'Anonymous'

                    # Convert the in-memory file to a Django ContentFile
                    image_file_for_db = ContentFile(in_memory_file, name=uploaded_file.name)

                    # Create and save the new database record
                    EmotionRecord.objects.create(
                        image=image_file_for_db,
                        username=form_username, 
                        emotion=predicted_label,
                        confidence=confidence_score
                    )
                    print(f"Prediction saved: {predicted_label} ({confidence_score:.2f})")
            except Exception as e:
                print(f"--- ERROR PROCESSING IMAGE ---")
                print(f"Error: {e}")
                context['prediction'] = f'Error processing image: {e}'
        else:
            print("--- NO FILE FOUND IN REQUEST ---")

    return render(request, 'emotion_detector/index.html', context)