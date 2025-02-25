import os
import subprocess

def convert_avi_to_mp4(directory):
    """Convert all .avi files in the specified directory to .mp4 using ffmpeg."""
    for filename in os.listdir(directory):
        if filename.lower().endswith(".avi"):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, os.path.splitext(filename)[0] + ".mp4")
            
            # Run ffmpeg command to copy streams without re-encoding
            command = [
                "ffmpeg", "-i", input_path, "-c:v", "copy", "-c:a", "copy", output_path
            ]
            
            print(f"Converting: {input_path} -> {output_path}")
            subprocess.run(command, check=True)

            print(f"Conversion completed: {output_path}")

if __name__ == "__main__":
    directory =  'data/thal/Videos'# Use current working directory
    convert_avi_to_mp4(directory)
