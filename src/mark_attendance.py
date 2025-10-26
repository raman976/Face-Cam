import pandas as pd
from datetime import datetime
import os

def mark_attendance(students, save_folder='data/attendance'):
    os.makedirs(save_folder, exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    time_str = datetime.now().strftime('%H:%M:%S')
    file_path = os.path.join(save_folder, f'attendance_{date_str}.csv')

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=['Name', 'Time'])

    for student in students:
        if student != "Unknown" and student not in df['Name'].values:
            df = pd.concat([df, pd.DataFrame({'Name':[student], 'Time':[time_str]})], ignore_index=True)

    df.to_csv(file_path, index=False)
    print(f"Attendance marked for {len([s for s in students if s!='Unknown'])} students.")
