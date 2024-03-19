import cv2
from google.colab.patches import cv2_imshow

# Load the input image
image_path = "/content/drive/MyDrive/CaseStudy_FontDetection_Sample/CaseStudy_FontDetection_Sample/DulcimerJF_24.png"
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to obtain a binary image
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Define ROI coordinates
roi_startX, roi_startY, roi_endX, roi_endY = 40, 80, 180, 40

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and aspect ratio to identify potential text regions
min_area = 50
min_aspect_ratio = 1.5
text_regions = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect_ratio = w / h
    if area > min_area and aspect_ratio > min_aspect_ratio:
        text_regions.append((x, y, x + w, y + h))

# Draw bounding boxes around the detected text regions that intersect with ROI
for (startX, startY, endX, endY) in text_regions:
    if startX >= roi_startX and startY >= roi_startY and endX <= roi_endX and endY <= roi_endY:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Draw ROI rectangle
cv2.rectangle(image, (roi_startX, roi_startY), (roi_endX, roi_endY), (255, 0, 0), 2)

# Display the result
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
