import cv2

def plot_labels_on_video(input_video_path, labels_txt_path1, labels_txt_path2, output_video_path, font_scale=1, font_thickness=2):
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Read labels from the text files
    with open(labels_txt_path1, 'r') as f1, open(labels_txt_path2, 'r') as f2:
        labels1 = f1.read().splitlines()[:-1]
        labels2 = f2.read().splitlines()[:-1]

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check label count matches frame count
    if len(labels1) != total_frames - 1 or len(labels2) != total_frames - 1:
        print(len(labels1), len(labels2), total_frames-1)
        print(f"Error: The number of labels in the files does not match the number of frames in the video.")
        return

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position1 = (50, 50)  # Position for the first set of labels (x, y)
    text_position2 = (50, 100)  # Position for the second set of labels (x, y)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay the labels on the frame
        if frame_index == total_frames - 1:
            break
        label1 = 'Pred: ' + labels1[frame_index]
        label2 = 'GT: ' + labels2[frame_index]

        cv2.putText(frame, label1, text_position1, font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
        cv2.putText(frame, label2, text_position2, font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Plotted video saved at {output_video_path}")

# Example usage
input_video = 'data/thal/Videos/IMG_4579_counter.mp4'
labels_txt1 = 'output/results/pred_gt_list/IMG_4579_counter-gt.txt'
labels_txt2 = 'output/results/pred_gt_list/IMG_4579_counter-pred.txt'
output_video = 'vis/IMG_4579_counter.mp4'
plot_labels_on_video(input_video, labels_txt1, labels_txt2, output_video)
