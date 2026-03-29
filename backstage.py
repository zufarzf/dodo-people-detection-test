import argparse
import cv2
import pandas as pd
from config import Config as conf




def parse_args():
    parser = argparse.ArgumentParser(description="Table monitoring prototype")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    return parser.parse_args()


def roi_to_xyxy(roi):
    x, y, w, h = roi
    return x, y, x + w, y + h


def boxes_intersect(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    if ax2 < bx1 or bx2 < ax1:
        return False

    if ay2 < by1 or by2 < ay1:
        return False

    return True


def select_table_roi(cap):
    ret, first_frame = cap.read()
    if not ret:
        print("Error: cannot read the first frame")
        return None, None

    # Ручной выбор столика — этого достаточно для прототипа из ТЗ
    roi = cv2.selectROI("Select table ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select table ROI")

    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        print("Error: ROI was not selected")
        return None, None

    table_box = roi_to_xyxy((x, y, w, h))
    return (x, y, w, h), table_box


def create_video_writer(cap, output_path):
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Error: cannot create output video writer")
        return None, None

    return writer, fps


def detect_people_near_table(model, frame, display_frame, table_box, confidence_threshold):
    person_near_table = False

    # YOLO запускаем на каждом кадре, чтобы получить bbox людей
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # В задаче нас интересуют только люди
            if cls_id != 0:
                continue

            if confidence < confidence_threshold:
                continue

            px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
            person_box = (px1, py1, px2, py2)

            intersects = boxes_intersect(table_box, person_box)

            if intersects:
                person_near_table = True
                color = (0, 0, 255)
                label = f"person {confidence:.2f} [near]"
            else:
                color = (255, 0, 0)
                label = f"person {confidence:.2f}"

            cv2.rectangle(display_frame, (px1, py1), (px2, py2), color, 2)
            cv2.putText(
                display_frame,
                label,
                (px1, max(py1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return person_near_table


def update_state(current_state, near_counter, empty_counter, person_near_table, frame_index, current_time_sec):
    event = None

    # Сглаживание по кадрам, чтобы состояние не дёргалось из-за шумных детекций
    if person_near_table:
        near_counter += 1
        empty_counter = 0
    else:
        empty_counter += 1
        near_counter = 0
    
    if current_state == "EMPTY" and near_counter >= conf.OCCUPIED_THRESHOLD:
        current_state = "OCCUPIED"
        event = {
            "frame": frame_index,
            "timestamp_sec": round(current_time_sec, 2),
            "event": "approach_detected",
            "state": current_state,
        }

    elif current_state == "OCCUPIED" and empty_counter >= conf.EMPTY_THRESHOLD:
        current_state = "EMPTY"
        event = {
            "frame": frame_index,
            "timestamp_sec": round(current_time_sec, 2),
            "event": "table_became_empty",
            "state": current_state,
        }

    return current_state, near_counter, empty_counter, event


def draw_table_and_hud(display_frame, roi, frame_index, current_time_sec, person_near_table,
                       current_state, near_counter, empty_counter):
    x, y, w, h = roi

    table_color = (0, 0, 255) if current_state == "OCCUPIED" else (0, 255, 0)

    # Цвет рамки столика показывает текущее состояние
    cv2.rectangle(display_frame, (x, y), (x + w, y + h), table_color, 2)

    cv2.putText(
        display_frame,
        f"Frame: {frame_index}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"Time: {current_time_sec:.2f}s",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"Raw near: {'YES' if person_near_table else 'NO'}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"State: {current_state}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        table_color,
        2
    )

    cv2.putText(
        display_frame,
        f"Near counter: {near_counter}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Empty counter: {empty_counter}",
        (20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Press \"q\" or \"Esc\" to exit",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


def build_delay_report(events_df):
    delay_rows = []

    empty_events = events_df[events_df["event"] == "table_became_empty"].reset_index(drop=True)
    approach_events = events_df[events_df["event"] == "approach_detected"].reset_index(drop=True)

    for _, empty_row in empty_events.iterrows():
        empty_time = empty_row["timestamp_sec"]
        next_approaches = approach_events[approach_events["timestamp_sec"] > empty_time]

        if next_approaches.empty:
            delay_rows.append({
                "empty_time_sec": empty_time,
                "next_approach_time_sec": None,
                "delay_sec": None,
            })
            continue

        next_approach_time = next_approaches.iloc[0]["timestamp_sec"]
        delay_sec = round(next_approach_time - empty_time, 2)

        delay_rows.append({
            "empty_time_sec": empty_time,
            "next_approach_time_sec": next_approach_time,
            "delay_sec": delay_sec,
        })

    return pd.DataFrame(delay_rows)


def save_reports(events):
    if not events:
        print("\nNo events collected.")
        return

    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values(by=["timestamp_sec", "frame"]).reset_index(drop=True)

    print("\nEvents DataFrame:")
    print(events_df)

    events_df.to_csv(conf.EVENTS_CSV_PATH, index=False)

    report_df = build_delay_report(events_df)

    print("\nDelay report:")
    print(report_df)

    report_df.to_csv(conf.REPORT_CSV_PATH, index=False)

    valid_delays = report_df["delay_sec"].dropna()

    if not valid_delays.empty:
        average_delay = round(valid_delays.mean(), 2)
        print(f"\nAverage delay: {average_delay:.2f} sec")
    else:
        print("\nAverage delay: not available (no valid pairs found)")

    print("\nSaved files:")
    print(f"- {conf.OUTPUT_VIDEO_PATH}")
    print(f"- {conf.EVENTS_CSV_PATH}")
    print(f"- {conf.REPORT_CSV_PATH}")