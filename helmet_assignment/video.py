import os
import cv2
import subprocess
from IPython.display import Video, display
import pandas as pd
import numpy as np


def video_with_predictions(
    video_path: str, sub_labels: pd.DataFrame, max_frame=9999, freeze_impacts=True, verbose=True
) -> str:
    """
    Annotates a video with both the baseline model boxes and ground truth boxes.
    """
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (0, 0, 0)  # Black

    INCORRECT_IMPACT_COLOR = (0, 0, 255)  # Red
    CORRECT_IMPACT_COLOR = (51, 255, 255)  # Yellow

    CORRECT_COLOR = (0, 255, 0)  # Green
    INCORRECT_COLOR = (255, 255, 255)  # White
    WHITE = (255, 255, 255)  # White

    video_name = os.path.basename(video_path).replace(".mp4", "")
    if verbose:
        print(f"Running for {video_name}")
    sub_labels = sub_labels.copy()
    # Add frame and video columns:
    sub_labels['video'] = sub_labels['video_frame'].str.split('_').str[:3].str.join('_')
    sub_labels['frame'] = sub_labels['video_frame'].str.split('_').str[-1].astype('int')

    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = "pred_" + video_name + ".mp4"
    tmp_output_path = "tmp_" + output_path
    output_video = cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1

        img_name = f"{frame} : {video_name}"
        cv2.putText(img, img_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, thickness=1)

        cv2.putText(img, str(frame), (1230, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, thickness=1)
        # Get stats about current state in frame
        stats = sub_labels.query('video == @video_name and frame <= @frame')
        correct_nonimp = len(stats.query('weight == 1 and isCorrect'))
        total_nonimp = len(stats.query('weight == 1'))
        correct_imp = len(stats.query('weight > 1 and isCorrect'))
        total_imp = len(stats.query('weight > 1'))
        correct_weighted = correct_nonimp + (correct_imp * 1000)
        total_weighted = total_nonimp + (total_imp * 1000)
        acc_imp = correct_imp / np.max([1, total_imp])
        acc_nonimp = correct_nonimp / np.max([1, total_nonimp])
        acc_weighted = correct_weighted / np.max([1, total_weighted])
        cv2.putText(
            img,
            f'{acc_imp:0.4f} Impact Boxes Accuracy :      ({correct_imp}/{total_imp})',
            (5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            thickness=1,
        )

        cv2.putText(
            img,
            f'{acc_nonimp:0.4f} Non-Impact Boxes Accuracy: ({correct_nonimp}/{total_nonimp})',
            (5, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            thickness=1,
        )

        cv2.putText(
            img,
            f'{acc_weighted:0.4f} Weighted Accuracy:     ({correct_weighted}/{total_weighted})',
            (5, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            thickness=1,
        )

        video_frame = f'{video_name}_{frame}'
        boxes = sub_labels.query("video_frame == @video_frame")
        if len(boxes) == 0:
            return
        for box in boxes.itertuples(index=False):
            if box.isCorrect and box.weight == 1:
                # CORRECT
                box_color = CORRECT_COLOR
                gt_color = CORRECT_COLOR
                pred_thickness = 1
            elif box.isCorrect and box.weight > 1:
                box_color = CORRECT_IMPACT_COLOR
                gt_color = CORRECT_IMPACT_COLOR
                pred_thickness = 3
            elif (box.isCorrect == False) and (box.weight > 1):
                box_color = INCORRECT_IMPACT_COLOR
                gt_color = INCORRECT_IMPACT_COLOR
                pred_thickness = 3
            elif (box.isCorrect == False) and (box.weight == 1):
                box_color = INCORRECT_COLOR
                gt_color = HELMET_COLOR
                pred_thickness = 1

            # Ground Truth Box
            cv2.rectangle(
                img,
                (box.left_gt, box.top_gt),
                (box.left_gt + box.width_gt, box.top_gt + box.height_gt),
                gt_color,
                thickness=1,
            )
            # Prediction Box
            cv2.rectangle(
                img,
                (int(box.left_sub), int(box.top_sub)),
                (int(box.left_sub + box.width_sub), int(box.top_sub + box.height_sub)),
                box_color,
                thickness=pred_thickness,
            )

            cv2.putText(
                img,
                f"{box.label_gt}:{box.label_sub}",
                (max(0, box.left_gt - box.width_gt), max(0, box.top_gt - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                WHITE,
                thickness=1,
            )

        if boxes['weight'].sum() > 22 and freeze_impacts:
            for _ in range(60):
                # Freeze for 60 frames on impacts
                output_video.write(img)
        else:
            output_video.write(img)

        if frame >= max_frame:
            break

    output_video.release()
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(
        ["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast", "-vcodec", "libx264", output_path]
    )
    os.remove(tmp_output_path)

    return output_path
