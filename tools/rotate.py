import os
import subprocess

def convert_and_rotate_videos(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".MOV"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base_name + ".mp4")
            
            if "counter" in filename.lower():
                rotate_filter = "transpose=1"  # Rotate 90 degrees clockwise
            elif "double" in filename.lower():
                rotate_filter = "transpose=2,transpose=2"  # Rotate 180 degrees
            else:
                rotate_filter = None  # No rotation
            
            # Build the ffmpeg command
            command = [
                "ffmpeg",
                "-i", input_path,
            ]
            
            if rotate_filter:
                command += ["-vf", rotate_filter]
            
            command += [
                "-c:v", "libx264",  # Use H.264 codec for output
                "-preset", "fast",  # Use a fast preset
                "-crf", "23",       # Set quality
                output_path
            ]
            
            # Run the command
            try:
                subprocess.run(command, check=True)
                print(f"Converted and rotated: {filename} -> {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_directory = "data/Colored Thal Dataset"
output_directory = "data/Colored Thal Dataset/rot"
convert_and_rotate_videos(input_directory, output_directory)
