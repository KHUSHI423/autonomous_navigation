import time
import cv2

from backend.camera import Camera
from backend.detector import ObjectDetector
from backend.path_planner import PathPlanner
from backend.controller import Controller
from dashboard.visualizer import Visualizer


def main():
    print("Starting Autonomous Navigation System")

    # -------- INITIALIZE COMPONENTS --------
    camera = Camera(camera_id=0, resolution=(320, 240), fps=30)
    detector = ObjectDetector(
        model_path="models/efficientdet_lite0.tflite",
        confidence_threshold=0.5,
    )
    planner = PathPlanner()
    controller = Controller()
    visualizer = Visualizer()

    # -------- START SYSTEM --------
    camera.initialize()
    detector.load_model()
    controller.initialize()

    print("System initialized. Press 'q' to quit.")

    prev_time = time.time()

    try:
        while True:
            # -------- READ CAMERA --------
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # -------- DETECTION --------
            detections = detector.detect(frame)

            # -------- PATH PLANNING --------
            decision = planner.plan(detections)

            # -------- CONTROL --------
            controller.execute_decision(decision)

            # -------- FPS CALC --------
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # -------- VISUALIZATION --------
            vis_frame = visualizer.visualize(
                frame=frame,
                detections=detections,
                decision=decision,
                fps=fps,
            )

            visualizer.show(vis_frame)

            # -------- EXIT --------
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received.")

    finally:
        print("Shutting down system...")
        camera.release()
        controller.shutdown()
        visualizer.cleanup()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
