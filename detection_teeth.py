import cv2
import numpy as np
import os 

class ImageProcessor:
    def __init__(self):
        pass

    def resize_image(self, image, width=400, height=400):
        return cv2.resize(image, (width, height))

    def stack_images_horizontally(self, image_list):
        # Get dimensions of the first image
        width = image_list[0].shape[1]
        height = image_list[0].shape[0]

        # Resize images to have the same height
        resized_images = [self.resize_image(img, height=height) for img in image_list]

        # Stack images horizontally
        stacked_image = np.hstack(resized_images)

        return stacked_image

    # Function to process images
    def process_images(self, image_paths):
        processed_images = []
        for path in image_paths:
            # Load the image
            image = cv2.imread(path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding to get binary image
            _, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
            median_brightness = np.median(gray[otsu_threshold == 255])
            threshold_value = int(median_brightness * 0.7)  # Adjust this multiplier as needed

            # Apply thresholding to create a binary image
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV )

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original image
            mask = np.ones_like(binary) * 255
            cv2.drawContours(mask, contours, -1, (0, 0, 0), cv2.FILLED)

            # Apply the mask to remove the areas of interest from the original image
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Clean the image to reduce noise
            cleaned_image = self.clean_image(masked_image)

            # Append the processed image to the list
            processed_images.append(cleaned_image)

        return processed_images

    # Function to crop tooth contour
    def crop_tooth_contour(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding to get binary image
        _, otsu_threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
        median_brightness = np.median(gray[otsu_threshold == 255])
        threshold_value = int(median_brightness * 0.7)  # Adjust this multiplier as needed

        # Apply thresholding to create a binary image
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV )

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming largest contour corresponds to the tooth boundary
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a mask for the tooth contour
            tooth_mask = np.zeros_like(gray)
            cv2.drawContours(tooth_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

            # Bitwise AND operation to keep the tooth area
            tooth_area = cv2.bitwise_and(image, image, mask=tooth_mask)

            # Find bounding box around tooth contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop the tooth area
            cropped_tooth = tooth_area[y:y+h, x:x+w]

            return cropped_tooth
        else:
            return None

    # Function to clean image (reduce noise)
    def clean_image(self, image):
        # Apply median blur to reduce noise
        cleaned_image = cv2.medianBlur(image, 5)
        return cleaned_image

    # Function to display stacked images
    def display_stacked_images(self, stacked_image):
        cv2.imshow('Stacked Images', stacked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def warning_message(self):
        # Display warning message
        warning_message = np.ones((100, 300), dtype=np.uint8) * 255
        cv2.putText(warning_message, "Click X for other images", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Warning", warning_message)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read_image(self):
        folder_path="images/"
        image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        return image_paths

if __name__ == "__main__":
    processor = ImageProcessor()
    
    image_paths = processor.read_image()

    # Display warning message before processing
    processor.warning_message()

    # Process images
    processed_images = processor.process_images(image_paths)
    
    # Stack processed images
    stacked_images = [processor.stack_images_horizontally([orig_img, proc_img]) 
                      for orig_img, proc_img in zip([cv2.imread(path) for path in image_paths], processed_images)]
    
    # Display stacked images
    for stacked_img in stacked_images:
        processor.display_stacked_images(stacked_img)
