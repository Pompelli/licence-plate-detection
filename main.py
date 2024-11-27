import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils

# Load the image
img = cv2.imread("archive/images/Cars309.png")
if img is None:
    print("Error: Image not found. Check the file path.")
    exit()

# Display original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply noise reduction
bfilter = cv2.bilateralFilter(gray, 4, 20, 20)
plt.imshow(bfilter, cmap='gray')
plt.title('Noise Reduced Image')
plt.show()


# Edge detection
edged = cv2.Canny(bfilter, 100, 150)
plt.imshow(edged, cmap='gray')
plt.title('Edge Detection')
plt.show()

kernel = np.ones((2, 2), np.uint8) 
dilated_image = cv2.dilate(edged, kernel, iterations=1)

plt.imshow(dilated_image, cmap='gray')
plt.title('Edges dialated')
plt.show()

# Find contours
keypoints = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Loop through contours to find a 4-point contour
min_area = 900  # You can tweak this threshold
location = None
for contour in contours:
    if cv2.contourArea(contour) > min_area:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            location = approx
            break

if location is None:
    print("Error: No contour with 4 vertices found.")
    exit()

# Display contours for debugging
contour_img = img.copy()
cv2.drawContours(contour_img, [location], -1, (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Contour Detection')
plt.show()

# Create a mask
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [location], -1, 255, -1)

# Apply mask to the image
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.title('Masked Image')
plt.show()


