
import face_recognition
import pickle
import numpy as np


try:
    with open('embeddings/embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    known_face_encodings = data.get('embeddings', [])
    known_face_names = data.get('names', [])
except Exception as e:
    print(f"Error loading embeddings: {e}")
    known_face_encodings = []
    known_face_names = []

def recognize_faces(image_path, threshold=0.5):

    try:
        with open('embeddings/embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        known_face_encodings = data.get('embeddings', [])
        known_face_names = data.get('names', [])
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        known_face_encodings = []
        known_face_names = []

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model='hog') 
    face_encodings = face_recognition.face_encodings(image, face_locations)

    print(f"DEBUG: {len(face_encodings)} faces detected in seminar image {image_path}")
    present_students = []

    if not known_face_encodings or not face_encodings:
        print("DEBUG: No known embeddings or no faces detected.")
        return present_students

    for i, encoding in enumerate(face_encodings):
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        print(f"DEBUG: Face {i+1} distances: {distances}")
        if len(distances) == 0:
            present_students.append("Unknown")
            continue
        min_distance = np.min(distances)
        print(f"DEBUG: Face {i+1} min_distance: {min_distance}")
        if min_distance < threshold:
            index = np.argmin(distances)
            name = known_face_names[index].split('_')[0]
            print(f"DEBUG: Face {i+1} matched name: {name}")
            present_students.append(name)
        else:
            print(f"DEBUG: Face {i+1} not recognized (distance too high)")
            present_students.append("Unknown")
    return present_students
