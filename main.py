import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

# Define a kernel for (dilation/erosion)
kernel = np.ones((2, 2), np.uint8)  
                 #2 x 2 kernel 

# Load the image
img = cv2.imread("archive/images/Cars311.png")
if img is None:  # Check if the image was loaded successfully
    print("Error: Image not found. Check the file path.")
    exit()

# Display the original image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR (OpenCV format) to RGB (Matplotlib format)
plt.title('Original Image')
plt.show()

# Convert the image to grayscale to simplify further processing
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter for noise reduction
bfilter = cv2.bilateralFilter(gray, 4, 20, 20)  
# - First parameter: Grayscale image
# - Second parameter: Diameter of the filter
# - Third/Fourth parameter: Color and spatial filtering strength (preserves edges)

# Enhance contrast for better OCR results
enhanced_img = cv2.convertScaleAbs(bfilter, alpha=1.5, beta=0)  
# - alpha: Increases contrast
# - beta: Keeps brightness unchanged

# Apply thresholding
_, threshold_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
# - Automatically determines threshold using Otsu's method
# - Pixels above the threshold are set to white (255), below to black (0)

plt.imshow(threshold_img, cmap='gray')
plt.title('Enhanced Image with Thresholding')  
plt.show()

# Apply dilation to close small gaps
dialated = cv2.dilate(threshold_img, kernel, iterations=1)  

plt.imshow(dialated, cmap='gray')
plt.title('Dilated Image')
plt.show()

# Perform edge detection using the Canny algorithm
edged = cv2.Canny(dialated, 100, 150)  
# - Thresholds 100 and 150 control sensitivity

# Further dilation to enhance the edges
dilated_image = cv2.dilate(edged, kernel, iterations=1)  

plt.imshow(dilated_image, cmap='gray')
plt.title('Edges Dilated')
plt.show()

# Find contours in the image
keypoints = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
# - RETR_TREE: Retrieves hierarchical contours
# - CHAIN_APPROX_SIMPLE: Stores only essential points of contours

contours = imutils.grab_contours(keypoints)  
# - Extract contours (platform-independent)

contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  
# - Sort contours by size (descending), keep the top 10

# Search for a 4-point contour (likely a rectangle)
min_area = 200  # Minimum area for valid contours
location = None  
for contour in contours:
    if cv2.contourArea(contour) > min_area:  # Check if the area is large enough
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)  
        # - arcLength: Perimeter of the contour

        if len(approx) == 4:  # Check if contour has 4 corners
            location = approx
            break

if location is None:  # Handle error if no suitable contour is found
    print("Error: No contour with 4 vertices found.")
    exit()

# Draw the found contour for visualization
contour_img = img.copy()  
cv2.drawContours(contour_img, [location], -1, (0, 255, 0), 3)  
# - location: Contour coordinates
# - Color: Green (0, 255, 0)
# - Thickness: 3 pixels

plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Contour Detection')
plt.show()

# Create a mask for the license plate region
mask = np.zeros(gray.shape, np.uint8)  # Create a black image with the same size as the original
cv2.drawContours(mask, [location], -1, 255, -1)  
# - Fills the mask within the detected contour with white (255)

# Extract the license plate region using the mask
new_image = cv2.bitwise_and(img, img, mask=mask)  
# - bitwise_and: Overlay the original image with the mask

# Get a rectangular bounding box for clean cropping
x, y, w, h = cv2.boundingRect(location)  
# - boundingRect: Finds a rectangle enclosing the contour
license_plate = new_image[y:y+h, x:x+w]  # Crop the license plate region

# Resize the region for better OCR performance
license_plate_resized = cv2.resize(license_plate, (800, 200))  
# - New dimensions: 800x200 pixels (OCR requires clear, readable images)

# Perform OCR to recognize text
reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English characters
result = reader.readtext(license_plate_resized)  
# - Reads text from the resized image

# Display OCR results and annotate detected text
for detection in result:
    text = detection[1]  # Extract detected text
    print(f"Detected text: {text}")  
    # Draw a rectangle around the detected text
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    cv2.rectangle(license_plate_resized, top_left, bottom_right, (0, 255, 0), 2)  
    # - Rectangle around the detected text
    cv2.putText(license_plate_resized, text, (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  
    # - Add text above the rectangle

# Display the result with the detected text
plt.imshow(cv2.cvtColor(license_plate_resized, cv2.COLOR_BGR2RGB))
plt.title('Detected Symbols (License Plate)')
plt.show()
