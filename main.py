import cv2
import numpy as np

reference_points = []

def capture_live_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    print("Press 'c' to capture an image of the room...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Please try again.")
            continue
        cv2.imshow('Room Capture', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('room_image.jpg', frame)
            print("Image captured successfully!")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return 'room_image.jpg'

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    print(f"Processed image shape: {img.shape}, Edges detected: {np.sum(edges > 0)}")
    return img, edges

def detect_corners(edges):
    corners = cv2.goodFeaturesToTrack(edges, 100, 0.01, 10)
    if corners is not None:
        corners = corners.reshape(-1, 2).astype(int)
        print(f"Detected {len(corners)} corners")
    else:
        corners = np.array([])
        print("No corners detected")
    return corners

def click_event(event, x, y, flags, param):
    global reference_points
    if event == cv2.EVENT_LBUTTONDOWN:
        reference_points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Set Reference', param)
        if len(reference_points) == 2:
            cv2.destroyWindow('Set Reference')

def set_reference(img):
    global reference_points
    reference_points = []
    cv2.imshow('Set Reference', img)
    cv2.setMouseCallback('Set Reference', click_event, img)
    print("Click on two points that are 3 feet apart in the image")
    while len(reference_points) < 2:
        cv2.waitKey(1)
    return reference_points

def calculate_pixels_per_foot(points):
    distance_pixels = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
    return distance_pixels / 3  # Assuming the points are 3 feet apart

def calculate_measurements(corners, pixels_per_foot):
    distances = []
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            distance_pixels = np.linalg.norm(corners[i] - corners[j])
            distance_feet = distance_pixels / pixels_per_foot
            distances.append((distance_feet, tuple(corners[i]), tuple(corners[j])))
    print(f"Calculated distances between {len(corners)} detected corners")
    return sorted(distances, key=lambda x: x[0], reverse=True)

def analyze_dimensions(measurements):
    if len(measurements) < 3:
        return [(f"Dimension {i+1}", m[0]) for i, m in enumerate(measurements)]
    
    longest = measurements[0]
    second_longest = measurements[1]
    third_longest = measurements[2]
    
    # Check if we have a right angle between the two longest dimensions
    vec1 = np.array(longest[1]) - np.array(longest[2])
    vec2 = np.array(second_longest[1]) - np.array(second_longest[2])
    dot_product = np.dot(vec1, vec2)
    magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if magnitudes != 0:
        cos_angle = dot_product / magnitudes
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        
        if 80 < angle_deg < 100:  # Roughly perpendicular (allowing for some error)
            return [
                ("Estimated Length", longest[0]),
                ("Estimated Width", second_longest[0]),
                ("Estimated Height", third_longest[0]),
                ("Largest Diagonal", max(longest[0], second_longest[0], third_longest[0]))
            ]
    
    # If not perpendicular or if there was an issue with angle calculation
    return [
        ("Longest dimension", longest[0]),
        ("Second longest", second_longest[0]),
        ("Third longest", third_longest[0]),
        ("Shortest detected", measurements[-1][0])
    ]

def visualize_results(img, corners, measurements):
    for corner in corners:
        x, y = corner
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow('Detected Features', img)
    print("Displaying detected features. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if measurements:
        print(f"Number of measurements: {len(measurements)}")
        print(f"First measurement: {measurements[0]}")

        
        analyzed_dimensions = analyze_dimensions(measurements)
        print("\nAnalyzed dimensions:")
        print(analyzed_dimensions)
        
        print("\nEstimated room measurements:")
        for name, value in analyzed_dimensions:
            print(f"{name}: {value:.2f} feet")
            print('final poutput')
        
        print("\nNote:")
        print("- These measurements are estimates based on the longest detected distances between corners.")
        print("- The actual room dimensions may be smaller due to furniture or other objects in the room.")
        print("- 'Estimated Height' might not be accurate if the ceiling is not visible in the image.")
        print("- For more accurate results, ensure the image captures full walls and corners clearly.")
    else:
        print("No measurements could be calculated.")

def main():
    try:
        image_path = capture_live_image()
        img, edges = process_image(image_path)
        corners = detect_corners(edges)
        
        if len(corners) < 2:
            print("Not enough corners detected to calculate measurements.")
            print("Try capturing the image again with more distinct features in view.")
            cv2.imshow('Processed Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        
        reference_points = set_reference(img)
        pixels_per_foot = calculate_pixels_per_foot(reference_points)
        
        measurements = calculate_measurements(corners, pixels_per_foot)
        print("Measurements calculated. Proceeding to visualize results...")
        visualize_results(img, corners, measurements)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()