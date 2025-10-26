import streamlit as st
import os
from datetime import datetime
import pandas as pd
import io
import shutil
import pathlib
import uuid

from generate_embeddings import generate_embeddings
from recognize_attendance import recognize_faces  


ROOT = pathlib.Path(__file__).resolve().parents[1]  
DATA_DIR = ROOT / "data"
STUDENTS_DIR = DATA_DIR / "students"
SEMINAR_PHOTOS_DIR = DATA_DIR / "seminar_photos"
ATTENDANCE_DIR = DATA_DIR / "attendance"
EMBEDDINGS_DIR = ROOT / "embeddings"


for p in (STUDENTS_DIR, SEMINAR_PHOTOS_DIR, ATTENDANCE_DIR, EMBEDDINGS_DIR):
    os.makedirs(p, exist_ok=True)

st.set_page_config(page_title="Attendance Uploader", layout="wide")

st.title("FACE CAM")



st.header("Student Registration")


uploaded_students = st.file_uploader("Upload student image files", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if st.button("(re)generate embeddings"):
    if not uploaded_students:
        st.warning("No student images uploaded.")
    else:
        saved = []
        new_names = set()
        for f in uploaded_students:
            fname = f.name
            base, ext = os.path.splitext(fname)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_suffix = uuid.uuid4().hex[:6]
            safe_name = f"{base}_{timestamp}_{unique_suffix}{ext}"
            dest_path = STUDENTS_DIR / safe_name
            with open(dest_path, "wb") as out:
                out.write(f.read())
            saved.append(dest_path.name)

            new_names.add(base.split('_')[0])

        names_path = ROOT / "student_names.txt"
        with open(names_path, "a") as names_file:
            for name in new_names:
                names_file.write(name + "\n")
        st.success(f"Saved {len(saved)} student images to `{STUDENTS_DIR}`.")
        st.info("Regenerating embeddings now (this may take some seconds on CPU)...")

        try:
            generate_embeddings(student_folder=str(STUDENTS_DIR), save_path=str(EMBEDDINGS_DIR / "embeddings.pkl"))
            st.success("Embeddings regenerated and saved to `embeddings/embeddings.pkl`.")
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")

st.header("Upload Seminar Photos")
st.write("Give the seminar a unique name")

seminar_name = st.text_input("Seminar name (no slashes or special characters)", value=f"seminar_{datetime.now().strftime('%Y%m%d')}")
uploaded_seminar_files = st.file_uploader("Upload seminar photos (multiple allowed)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if st.button("attendance file"):
    if not seminar_name.strip():
        st.warning("Please provide a seminar name.")
    elif not uploaded_seminar_files:
        st.warning("Please upload at least one seminar photo.")
    else:
        # sanitize seminar filename
        safe_seminar = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in seminar_name).strip().replace(" ", "_")
        attendance_filename = f"attendance_{safe_seminar}.csv"
        attendance_path = ATTENDANCE_DIR / attendance_filename


        per_image_logs = []
        present_set = set()
        processed_images = []

        with st.spinner("Saving seminar photos and scanning (this may take some seconds per image on CPU)..."):
            # Save uploaded photos into seminar_photos with unique names
            for f in uploaded_seminar_files:
                orig_name = f.name
                base, ext = os.path.splitext(orig_name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_name = f"{base}_{timestamp}_{uuid.uuid4().hex[:6]}{ext}"
                save_path = SEMINAR_PHOTOS_DIR / unique_name
                with open(save_path, "wb") as out:
                    out.write(f.read())
                processed_images.append(save_path)
            
            # For each saved image, call your recognize_faces function
            for img_path in processed_images:
                try:
                    detected = recognize_faces(str(img_path), threshold=0.5)
                except TypeError:
                    detected = recognize_faces(str(img_path))
                except Exception as e:
                    st.error(f"Error processing image {img_path.name}: {e}")
                    detected = []

                detected_known = [d for d in detected if d != "Unknown"]
                for name in detected_known:
                    present_set.add(name)
                per_image_logs.append({"image": img_path.name, "detected": detected_known})

        # Build attendance dataframe
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        first_seen = {}
        for log in per_image_logs:
            img = log["image"]
            for name in log["detected"]:
                if name not in first_seen:
                    first_seen[name] = img
        for name in sorted(present_set):
            rows.append({
                "Name": name,
                "FirstDetectedInImage": first_seen.get(name, ""),
                "Seminar": seminar_name,
                "TimeDetected": now_str
            })
        df_att = pd.DataFrame(rows, columns=["Name", "FirstDetectedInImage", "Seminar", "TimeDetected"])
        if attendance_path.exists():
            df_existing = pd.read_csv(attendance_path)
            df_att = pd.concat([df_existing, df_att], ignore_index=True)
        # Remove duplicate student entries, keeping the first occurrence
        df_att = df_att.drop_duplicates(subset=["Name", "Seminar"], keep="first")
        df_att.to_csv(attendance_path, index=False)
        st.success(f"Attendance file saved: `{attendance_path}` — {len(df_att)} unique present students.")

        for img_path in processed_images:
            try:
                os.remove(img_path)
                st.info(f"Deleted seminar image: {img_path}")
            except Exception as e:
                st.warning(f"Could not delete seminar image {img_path}: {e}")

        st.subheader("Per-image detection log")
        rows2 = []
        for log in per_image_logs:
            rows2.append({"Image": log["image"], "DetectedNames": ", ".join(log["detected"]) if log["detected"] else "(none)"})
        df_logs = pd.DataFrame(rows2)
        st.dataframe(df_logs)

        st.subheader("Aggregated attendance (one row per student)")
        st.dataframe(df_att)

        with open(attendance_path, "rb") as f:
            csv_bytes = f.read()
        st.download_button(label="Download attendance CSV", data=csv_bytes, file_name=attendance_filename, mime="text/csv")

