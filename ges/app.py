import cv2
import numpy as np
import math

# Function to calculate the angle between fingers
def calculate_angle(far, start, end):
    a = np.linalg.norm(np.array(start) - np.array(far))
    b = np.linalg.norm(np.array(end) - np.array(far))
    c = np.linalg.norm(np.array(start) - np.array(end))
    angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2 * a * b)))
    return angle

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (which is most likely the hand)
        max_contour = max(contours, key=cv2.contourArea)

        # Find the convex hull
        hull = cv2.convexHull(max_contour, returnPoints=False)

        # Find convexity defects
        defects = cv2.convexityDefects(max_contour, hull)

        if defects is not None:
            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate the angle between the fingers
                angle = calculate_angle(far, start, end)

                # Detect fingers based on the angle and depth of convexity defects
                if angle < 90 and d > 10000:  # Adjust depth threshold for your hand
                    finger_count += 1
                    cv2.circle(frame, far, 5, [0, 0, 255], -1)  # Mark defect points

            # Display the number of fingers (finger_count + 1 for the thumb)
            cv2.putText(frame, f'Fingers: {finger_count + 1}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw contours and convex hull on the frame
        hull_points = cv2.convexHull(max_contour)
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull_points], -1, (0, 255, 255), 2)

    # Show the frames
    cv2.imshow('Hand Gesture', frame)

    # Exit on pressing 'l'
    if cv2.waitKey(1) & 0xFF == ord('l'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
