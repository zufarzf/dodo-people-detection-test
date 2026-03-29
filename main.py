import cv2
from ultralytics import YOLO

from backstage import (
    parse_args,
    select_table_roi,
    create_video_writer,
    detect_people_near_table,
    update_state,
    draw_table_and_hud,
    save_reports,
)
from config import Config as conf




def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: cannot open video file: {args.video}")
        return

    roi, table_box = select_table_roi(cap)
    if roi is None:
        cap.release()
        return

    # После выбора ROI возвращаемся в начало видео
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer, fps = create_video_writer(cap, conf.OUTPUT_VIDEO_PATH)
    if writer is None:
        cap.release()
        return

    # Используем лёгкую готовую модель YOLO для детекции людей
    model = YOLO("yolov8n.pt")

    current_state = "EMPTY"
    near_counter  = 0
    empty_counter = 0
    frame_index   = 0
    events        = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        current_time_sec = frame_index / fps if fps > 0 else 0

        person_near_table = detect_people_near_table(
            model=model,
            frame=frame,
            display_frame=display_frame,
            table_box=table_box,
            confidence_threshold=conf.CONFIDENCE_THRESHOLD,
        )

        current_state, near_counter, empty_counter, event = update_state(
            current_state=current_state,
            near_counter=near_counter,
            empty_counter=empty_counter,
            person_near_table=person_near_table,
            frame_index=frame_index,
            current_time_sec=current_time_sec,
        )

        if event:
            events.append(event)
            print(f"[EVENT] {current_time_sec:.2f}s -> {event['event']}")

        draw_table_and_hud(
            display_frame=display_frame,
            roi=roi,
            frame_index=frame_index,
            current_time_sec=current_time_sec,
            person_near_table=person_near_table,
            current_state=current_state,
            near_counter=near_counter,
            empty_counter=empty_counter,
        )

        writer.write(display_frame)
        cv2.imshow("YOLO table monitoring", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    save_reports(events)


if __name__ == "__main__":
    main()