import face_recognition
import os
import pickle
import numpy as np

def generate_embeddings(student_folder='data/students', save_path='embeddings/embeddings.pkl'):
    embeddings = []
    names = []
    for file in os.listdir(student_folder):
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(student_folder, file)
            print(f"Processing file: {image_path}")
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            # Extract original name before timestamp and suffix
            # Remove last two underscore-separated parts (timestamp and suffix)
            base_parts = os.path.splitext(file)[0].split('_')
            if len(base_parts) > 2:
                original_name = '_'.join(base_parts[:-2])
            else:
                original_name = os.path.splitext(file)[0]
            name_with_space = original_name.replace('_', ' ')
            if encodings:
                print(f"Face detected in {file}")
                for encoding in encodings:
                    embeddings.append(encoding)
                    names.append(name_with_space)
            else:
                print(f"No face detected in {file}")
            # Delete student image after processing
            try:
                os.remove(image_path)
                print(f"Deleted {image_path}")
            except Exception as e:
                print(f"Could not delete {image_path}: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'names': names}, f)
    if embeddings:
        print(f"Saved {len(embeddings)} embeddings for {len(set(names))} students.")
    else:
        print("Warning: No face embeddings detected. Empty file was saved.")

# Ensure the function runs and prints output when script is executed directly
if __name__ == "__main__":
    generate_embeddings()
