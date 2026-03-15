from face_run import run_emotion_detection as face_runner
from voice_run import runner as voice_runner
from moviepy.editor import VideoFileClip
import os
import subprocess

# Make sure ffmpeg path is set
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

def convert_to_mp4(input_file, output_file):
    """Convert WebM to MP4 using ffmpeg"""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file


def extract_audio(input_file, output_file):
    """Extract audio from video into mp3"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    video = VideoFileClip(input_file, fps_source="fps")
    audio = video.audio
    audio.write_audiofile(output_file)
    video.close()
    print(f"Audio successfully extracted to {output_file}")


def run(path):
    # Convert webm → mp4 if needed
    if path.endswith(".webm"):
        mp4_path = path.replace(".webm", ".mp4")
        convert_to_mp4(path, mp4_path)
        path = mp4_path

    # ---- Run face model ----
    face_dict = face_runner(path)
    face_value = max(face_dict, key=face_dict.get)

    # ---- Extract audio ----
    audio_dir = "audios"
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, os.path.splitext(os.path.basename(path))[0] + ".mp3")
    extract_audio(path, audio_path)

    # ---- Run voice model ----
    voice_value = voice_runner(audio_path)
    if isinstance(voice_value, list):   # if it returns ['happy']
        voice_value = voice_value[0]

    # ---- Final decision ----
    if face_value == voice_value:
        final_value = face_value
    else:
        # ⚠️ Adjust logic here depending on what you want
        # Option A: prefer face
        final_value = face_value
        # Option B: prefer audio
        # final_value = voice_value
        # Option C: always mark conflict
        # final_value = "Conflict"

    print(f"[DEBUG] File: {path} | Face: {face_value} | Audio: {voice_value} | Final: {final_value}")
    return face_value, voice_value, final_value


if __name__ == '__main__':
    for file in os.listdir('videos'):
        if file.endswith((".mp4", ".webm")):
            print(file, run(f'videos/{file}'))
