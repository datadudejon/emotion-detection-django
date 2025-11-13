# AI Emotion Detection Web App

This is a Django-based web application that uses a TensorFlow/Keras Convolutional Neural Network (CNN) to perform emotion detection from user-uploaded images.

This project fulfills the requirements for the assignment, including a full web backend, a trained ML model, database integration, and all specified project files.

**Student ID:** `ENYI_23AG032707`

---

## 🚀 Features

* **Image Upload:** Users can provide their name and upload an image (JPG, PNG, etc.).
* **Face Detection:** Uses an OpenCV Haar Cascade to automatically find and crop the face from the uploaded image.
* **Emotion Prediction:** The cropped face is fed into a trained CNN model, which predicts one of seven emotions:
    * Angry
    * Disgust
    * Fear
    * Happy
    * Neutral
    * Sad
    * Surprise
* **Database Logging:** Every successful prediction (including the username, original image, predicted emotion, and confidence score) is saved to the SQLite database.
* **Admin Panel:** A full Django admin interface (`/admin`) is set up to view, add, edit, and delete all prediction records.

---

## 🛠 Tech Stack

* **Backend:** Python, Django
* **Machine Learning:** TensorFlow (Keras)
* **Image Processing:** OpenCV, Pillow, NumPy
* **Database:** SQLite 3

---

## 🏁 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [your-github-repo-url]
    cd ENYI_23AG032707
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run Database Migrations**
    (This creates the `db.sqlite3` file and the `EmotionRecord` table)
    ```bash
    python manage.py migrate
    ```

5.  **Create a Superuser**
    (This creates your admin account for the `/admin` panel)
    ```bash
    python manage.py createsuperuser
    ```

6.  **Run the Application**
    ```bash
    python manage.py runserver
    ```

7.  **Access the App**
    * **Main App:** `http://localhost:8000/`
    * **Admin Panel:** `http://localhost:8000/admin/`