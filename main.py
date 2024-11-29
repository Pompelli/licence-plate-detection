import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr  # for text recognition

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

# Enhance the image for OCR (contrast and thresholding)
enhanced_img = cv2.convertScaleAbs(bfilter, alpha=1.5, beta=0)  # Increase contrast
_, threshold_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.imshow(threshold_img, cmap='gray')
plt.title('Enhanced Image with Thresholding')
plt.show()

# Edge detection
edged = cv2.Canny(threshold_img, 100, 150)
plt.imshow(edged, cmap='gray')
plt.title('Edge Detection')
plt.show()

# Dilation to enhance edges
kernel = np.ones((2, 2), np.uint8) 
dilated_image = cv2.dilate(edged, kernel, iterations=1)
plt.imshow(dilated_image, cmap='gray')
plt.title('Edges Dilated')
plt.show()

# Find contours
keypoints = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Loop through contours to find a 4-point contour
min_area = 900  
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

# Create a mask for the detected license plate region
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [location], -1, 255, -1)

# Apply mask to the image to extract the license plate region
new_image = cv2.bitwise_and(img, img, mask=mask)

# Crop the license plate region (optional: if mask is too large)
x, y, w, h = cv2.boundingRect(location)
license_plate = new_image[y:y+h, x:x+w]

# Resize the image for better OCR recognition
license_plate_resized = cv2.resize(license_plate, (800, 200))

# Use EasyOCR to read symbols (license plate)
reader = easyocr.Reader(['en'])  # Create an OCR reader object for English
result = reader.readtext(license_plate_resized)

# Display OCR results
for detection in result:
    text = detection[1]
    print(f"Detected text: {text}")
    # Draw the bounding box around the detected text
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    cv2.rectangle(license_plate_resized, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(license_plate_resized, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Show the result with detected text
plt.imshow(cv2.cvtColor(license_plate_resized, cv2.COLOR_BGR2RGB))
plt.title('Detected Symbols (License Plate)')
plt.show()

# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
# import imutils

# # Load the image
# img = cv2.imread("archive/images/Cars309.png")
# if img is None:
#     print("Error: Image not found. Check the file path.")
#     exit()

# # Display original image
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.show()

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply noise reduction
# bfilter = cv2.bilateralFilter(gray, 4, 20, 20)
# plt.imshow(bfilter, cmap='gray')
# plt.title('Noise Reduced Image')
# plt.show()


# # Edge detection
# edged = cv2.Canny(bfilter, 100, 150)
# plt.imshow(edged, cmap='gray')
# plt.title('Edge Detection')
# plt.show()

# kernel = np.ones((2, 2), np.uint8) 
# dilated_image = cv2.dilate(edged, kernel, iterations=1)

# plt.imshow(dilated_image, cmap='gray')
# plt.title('Edges dialated')
# plt.show()

# # Find contours
# keypoints = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(keypoints)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# # Loop through contours to find a 4-point contour
# min_area = 900  # You can tweak this threshold
# location = None
# for contour in contours:
#     if cv2.contourArea(contour) > min_area:
#         approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
#         if len(approx) == 4:
#             location = approx
#             break

# if location is None:
#     print("Error: No contour with 4 vertices found.")
#     exit()

# # Display contours for debugging
# contour_img = img.copy()
# cv2.drawContours(contour_img, [location], -1, (0, 255, 0), 3)
# plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
# plt.title('Contour Detection')
# plt.show()

# # Create a mask
# mask = np.zeros(gray.shape, np.uint8)
# cv2.drawContours(mask, [location], -1, 255, -1)

# # Apply mask to the image
# new_image = cv2.bitwise_and(img, img, mask=mask)
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# plt.title('Masked Image')
# plt.show()


