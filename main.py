import cv2
import argparse


def parse_args():
	parser = argparse.ArgumentParser(description="Table monitoring prototype")
	parser.add_argument(
		"--video",
		type     = str,
		required = True,
		help     = "Path to input video file"
	)
	return parser.parse_args()



def main():
	args = parse_args()

	cap = cv2.VideoCapture(args.video)

	if not cap.isOpened():
		print(f"Error:  cannot open video file: {args.video}")
		return
	
	fps          = cap.get(cv2.CAP_PROP_FPS)
	width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	duration_sec = frame_count / fps if fps > 0 else 0

	print("Video info:")
	print(f"	FPS: {fps}")
	print(f"	Width: {width}")
	print(f"	Height: {height}")
	print(f"	Total frames: {frame_count}")
	print(f"	Duration (sec): {duration_sec:.2f}")

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
	
	print(f"Selected ROI: {x=}, {y=}, {w=}, {h=}")

	cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

	frame_index = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		display_frame = frame.copy()

		cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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

		cv2.imshow("Video", display_frame)

		key = cv2.waitKey(20) & 0xFF
		if key == 27 or key == ord("q"):
			break

		frame_index += 1

	cap.release()
	cv2.destroyAllWindows()




if __name__ == "__main__":
	main()