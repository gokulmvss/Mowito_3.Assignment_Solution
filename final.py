import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load template and test images
template_img = cv2.imread('C:/Users/Gokul/Desktop/mowito/template_images/template_image_1.jpg')
test_img = cv2.imread('C:/Users/Gokul/Desktop/mowito/test_images/test_image_1.jpg')

# Convert images to grayscale
template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to template image
template_thresh = cv2.adaptiveThreshold(template_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply adaptive thresholding to test image
test_thresh = cv2.adaptiveThreshold(test_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in template image
template_contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find contours in test image
test_contours, _ = cv2.findContours(test_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding rectangles around contours
template_img_with_rect = template_img.copy()
test_img_with_rect = test_img.copy()

if template_contours:
    max_contour_template = max(template_contours, key=cv2.contourArea)
    x_t, y_t, w_t, h_t = cv2.boundingRect(max_contour_template)
    cv2.rectangle(template_img_with_rect, (x_t, y_t), (x_t + w_t, y_t + h_t), (0, 255, 0), 2)

if test_contours:
    max_contour_test = max(test_contours, key=cv2.contourArea)
    x_test, y_test, w_test, h_test = cv2.boundingRect(max_contour_test)
    cv2.rectangle(test_img_with_rect, (x_test, y_test), (x_test + w_test, y_test + h_test), (0, 255, 0), 2)



# Ensure only the largest contour is considered
template_contour = max(template_contours, key=cv2.contourArea)
test_contour = max(test_contours, key=cv2.contourArea)    

# Calculate moments for the largest contour in each image
template_moments = cv2.moments(template_contour)
test_moments = cv2.moments(test_contour)

# Calculate orientation using the moments
template_orientation = -0.5 * np.arctan2(2 * template_moments['mu11'], template_moments['mu20'] - template_moments['mu02']) * 180 / np.pi
test_orientation = -0.5 * np.arctan2(2 * test_moments['mu11'], test_moments['mu20'] - test_moments['mu02']) * 180 / np.pi

# Calculate rotation angle difference
rotation_angle = test_orientation - template_orientation




image1 = cv2.resize(template_img_with_rect, (500, 500))
image2 = cv2.resize(test_img_with_rect, (500, 500))

# Create a new image to display both images
combined_image = np.hstack((image1, image2))

# Display the combined image and the result
cv2.putText(combined_image, f"Rotation Angle: {rotation_angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('Template Image  vs Test Image With polygon boundary', combined_image)
cv2.waitKey(0)
#cv2.destroyAllWindows()


plt.figure(figsize=(20,4))
plt.subplot(1,3,1),plt.imshow(template_img_with_rect),plt.title("Template Image with Boundary")
plt.subplot(1,3,2),plt.imshow(test_img_with_rect),plt.title("Test image with Boundary")
print("Rotation angle:", rotation_angle)
plt.show()