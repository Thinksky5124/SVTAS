import cv2

def plot_labels_on_video(input_video_path, gt_txt_path1, gt_txt_path2, pred_txt_path1, pred_txt_path2, output_video_path, font_scale=1, font_thickness=2):
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Read and concatenate labels from the text files
    with open(gt_txt_path1, 'r') as gt1, open(gt_txt_path2, 'r') as gt2:
        gt_labels1 = gt1.read().splitlines()[:-1]
        gt_labels2 = gt2.read().splitlines()[:-1]
        gt_labels = [f"{g1} {g2}" for g1, g2 in zip(gt_labels1, gt_labels2)]

    with open(pred_txt_path1, 'r') as pred1, open(pred_txt_path2, 'r') as pred2:
        pred_labels1 = pred1.read().splitlines()[:-1]
        pred_labels2 = pred2.read().splitlines()[:-1]
        pred_labels = [f"{p1} {p2}" for p1, p2 in zip(pred_labels1, pred_labels2)]

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check label count matches frame count
    if len(gt_labels) != total_frames - 1 or len(pred_labels) != total_frames - 1:
        print(len(gt_labels), len(pred_labels), total_frames - 1)
        print(f"Error: The number of labels in the files does not match the number of frames in the video.")
        return

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for AVI
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position_gt = (50, 50)  # Position for ground truth labels (x, y)
    text_position_pred = (50, 100)  # Position for prediction labels (x, y)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay the labels on the frame
        if frame_index == total_frames - 1:
            break
        gt_label = 'GT: ' + gt_labels[frame_index]
        pred_label = 'Pred: ' + pred_labels[frame_index]

        cv2.putText(frame, gt_label, text_position_gt, font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
        cv2.putText(frame, pred_label, text_position_pred, font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

        frame_index += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Plotted video saved at {output_video_path}")

# Example usage
input_video = 'data/thal/Videos/IMG_4742_counter.mp4'
gt_txt1 = 'output/results/pred_gt_list/IMG_4742_counter-b1-gt.txt'
gt_txt2 = 'output/results/pred_gt_list/IMG_4742_counter-tag-gt.txt'
pred_txt1 = 'output/results/pred_gt_list/IMG_4742_counter-b1-pred.txt'
pred_txt2 = 'output/results/pred_gt_list/IMG_4742_counter-tag-pred.txt'
output_video = 'vis/IMG_4742_counter.mp4'

plot_labels_on_video(input_video, gt_txt1, gt_txt2, pred_txt1, pred_txt2, output_video)
