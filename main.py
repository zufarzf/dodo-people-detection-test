import cv2
import argparse
from ultralytics import YOLO




def parse_args():
	parser = argparse.ArgumentParser(description="Table monitoring prototype")
	parser.add_argument(
		"--video",
		type     = str,
		required = True,
		help     = "Path to input video file"
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


def check_person_near_table_in_frame(model, frame, display_frame, table_box):
	results              = model(frame, verbose=False)
	person_near_table    = False
	confidence_threshold = 0.2

	for result in results:
		boxes = result.boxes

		for box in boxes:
			cls_id     = int(box.cls[0].item())
			confidence = float(box.conf[0].item())

			if cls_id != 0:
				continue

			if confidence < confidence_threshold:
				continue

			px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
			person_box = (px1, py1, px2, py2)

			intersects = boxes_intersect(table_box, person_box)

			if intersects:
				person_near_table = True
				color             = (0, 0, 255)
				label             = f"persone {confidence:.2f} [near]"
			else:
				color = (255, 0, 0)
				label = f"persone {confidence:.2f}"

			cv2.rectangle(
				display_frame,
				(px1, py1),
				(px2, py2),
				color,
				2
			)
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




def main():
	args = parse_args()

	cap = cv2.VideoCapture(args.video)

	if not cap.isOpened():
		print(f"Error:  cannot open video file: {args.video}")
		return
	
	fps = cap.get(cv2.CAP_PROP_FPS)
	
	ret, first_frame = cap.read()
	if not ret:
		print("Error: cannot read the first frame")
		cap.release()
		return
	
	roi = cv2.selectROI("Select table ROI", first_frame, fromCenter=False, showCrosshair=True)
	cv2.destroyWindow("Select table ROI")
	
	x, y, w, h = map(int, roi)

	if w == 0 or h == 0:
		print("Error: ROI was not selected")
		cap.release()
		return
	
	table_box = roi_to_xyxy((x, y, w, h))

	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	model = YOLO("yolov8n.pt")

	occupied_threshold = 8
	empty_threshold = 12

	current_state = "EMPTY"
	near_counter = 0
	empty_counter = 0

	frame_index = 0
	events = []

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		display_frame     = frame.copy()
		person_near_table = check_person_near_table_in_frame(model, frame, display_frame, table_box)

		if person_near_table:
			near_counter += 1
			empty_counter = 0

		else:
			empty_counter += 1
			near_counter   = 0


		if current_state == "EMPTY" and near_counter >= occupied_threshold:
			current_state = "OCCUPIED"
			events.append({
				"frame": frame_index,
				"timestamp_sec": round(current_time_sec, 2),
				"event": "approach_detected",
				"state": current_state
			})
			print(f"[EVENT] {current_time_sec:.2f}s -> approach_detected")
		
		elif current_state == "OCCUPIED" and empty_counter >= empty_threshold:
			current_state = "EMPTY"
			events.append({
				"frame": frame_index,
				"timestamp_sec": round(current_time_sec, 2),
				"event": "table_became_empty",
				"state": current_state
			})
			print(f"[EVENT] {current_time_sec:.2f}s -> table_became_empty")


		table_color = (0, 0, 255) if current_state == "OCCUPIED" else (0, 255, 0)
		
		cv2.rectangle(
			display_frame,
			(x, y),
			(x + w, y + h),
			table_color,
			2
		)

		current_time_sec = frame_index / fps if fps > 0 else 0

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

		cv2.imshow("YOLO table monitoring", display_frame)

		key = cv2.waitKey(20) & 0xFF
		if key == 27 or key == ord("q"):
			break

		frame_index += 1

	cap.release()
	cv2.destroyAllWindows()

	print("\nCollected events:")
	for event in events:
		print(event)




if __name__ == "__main__":
	main()