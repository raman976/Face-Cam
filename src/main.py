from recognize_attendance import recognize_faces
from mark_attendance import mark_attendance

# Photo captured via phone/selfie stick
image_path = 'data/seminar_photos/seminar2.jpg'

students_present = recognize_faces(image_path, threshold=0.5)
print("Present Students:", students_present)

mark_attendance(students_present)

# Delete seminar image after processing
import os
try:
	os.remove(image_path)
	print(f"Deleted seminar image: {image_path}")
except Exception as e:
	print(f"Could not delete seminar image {image_path}: {e}")
