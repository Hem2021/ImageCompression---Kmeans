# Import necessary libraries
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import flask
from flask import Flask, request, render_template, send_file
import io
import tempfile
import math

# Flask Setup
app = Flask(__name__)

# Function to Apply K-means Clustering to Reduce Colors
def apply_kmeans(pixels, k):
    # K-means clustering using sklearn
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Assign each pixel the color of its respective centroid
    compressed_pixels = centroids[labels].astype('uint8')
    return compressed_pixels

# Function to Compress Image and Convert to Indexed Format
def compress_image(image_path, k):
    # Open the image
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format

    # Convert image to numpy array and reshape
    pixels = np.array(image)
    original_shape = pixels.shape
    pixels = pixels.reshape(-1, 3)  # Reshape to (num_pixels, 3)

    # Apply K-means to reduce colors
    compressed_pixels = apply_kmeans(pixels, k)

    # Reshape back to original image dimensions
    compressed_pixels = compressed_pixels.reshape(original_shape)
    compressed_image = Image.fromarray(compressed_pixels, 'RGB')

    # Convert the compressed image to indexed color palette
    compressed_image = compressed_image.convert("P", palette=Image.ADAPTIVE, colors=k)

    return compressed_image

# Function to Calculate Image Size Based on Type
def calculate_image_size(image):
    # Calculate size in bytes depending on image mode
    if image.mode == "RGB":
        # For RGB images, each pixel is represented by 3 bytes
        return image.width * image.height * 3
    elif image.mode == "P":
        # For indexed images: palette size + indexed pixel data
        num_unique_colors = len(image.getpalette()) // 3
        palette_size = num_unique_colors * 3  # 3 bytes per color in the palette
        bits_per_pixel = math.ceil(math.log2(num_unique_colors))
        indexed_data_size = (image.width * image.height * bits_per_pixel) / 8  # Convert bits to bytes
        return palette_size + int(indexed_data_size)
    else:
        return len(np.array(image).flatten())

# Flask Route for Home Page
@app.route('/')
def home():
    return render_template('index.html', show_details=False)

# Flask Route for Image Upload and Compression
@app.route('/compress', methods=['POST'])
def compress():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    # Handle edge cases for unsupported file types
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('index.html', error="Unsupported file type. Please upload a PNG or JPEG image.")

    # Convert file to a temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        try:
            k = int(request.form.get('colors', 16))  # Default to 16 colors if not specified
            
            # Open original image and convert to PNG format
            original_image = Image.open(temp_file.name).convert('RGB')
            original_image.save(temp_file.name, format="PNG")

            # Calculate original image size
            original_size = calculate_image_size(original_image)

            # Compress the image
            compressed_image = compress_image(temp_file.name, k)

            # Calculate compressed image size
            compressed_size = calculate_image_size(compressed_image)

            # Save compressed image to an in-memory buffer
            buffer = io.BytesIO()
            compressed_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Save buffer globally to be used in the download route
            global compressed_buffer
            compressed_buffer = buffer

            # Display original vs compressed size
            size_reduction = ((original_size - compressed_size) / original_size) * 100

            # Send the compressed image back to the user with size comparison
            return render_template('index.html', success=True, show_details=True, 
                                   download_link='/download_compressed', 
                                   original_size=f"{round(original_size / 1024, 2)} KB", 
                                   compressed_size=f"{round(compressed_size / 1024, 2)} KB", 
                                   size_reduction=f"{size_reduction:.2f}%")
        finally:
            # Remove temporary file
            os.unlink(temp_file.name)

# Flask Route for Downloading Compressed Image
@app.route('/download_compressed', methods=['GET'])
def download_compressed():
    # Retrieve the compressed image from the buffer
    if 'compressed_buffer' in globals() and compressed_buffer:
        compressed_buffer.seek(0)
        return send_file(compressed_buffer, mimetype='image/png', as_attachment=True, download_name='compressed_image.png')
    else:
        return "No compressed image available."

# Flask Application Run
# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    from gunicorn.app.wsgiapp import run
    run()


"""
Explanation:
1. **Flask Web App**: Provides a user interface where users can upload their images for compression.
2. **Image Compression with K-means**: Reduces the number of colors using scikit-learn's K-means implementation. Default is set to 16 colors.
3. **Indexed Color Conversion**: Converts the reduced-color image into an indexed color format to further save space.
4. **Direct Image Download**: After compression, users can directly download the image without the need for AWS S3 storage.
5. **Edge Case Handling**: Handles unsupported file types, missing files, and temporary file removal for cleanups.
6. **Image Size Calculation**: Calculates the size of the original and compressed images to display size reduction percentage.
7. **Frontend**: The HTML template (`index.html`) is enhanced to provide a more user-friendly UI, displaying errors, success messages, and download links.

Technology Stack:
- **Python Libraries**: Flask (for web app), Pillow (for image processing), scikit-learn (for K-means clustering).
- **Frontend**: Simple HTML form for image upload, with success/error messages and download links for enhanced user experience.
"""


"""
### Detailed Explanation of Image Size Calculation Logic

The process of calculating the actual size of an image and its final compressed size is a key highlight of this project, demonstrating the efficiency of K-means clustering for image compression. Below is a detailed breakdown of the logic used:

1. **Original Image Size Calculation**:
   - For **RGB images**, each pixel is represented by three color channels: Red, Green, and Blue. Each channel requires **1 byte** of storage, leading to a total of **3 bytes per pixel**.
   - The formula for calculating the size of an RGB image is:
     
     ↳ **Size (in bytes) = Width × Height × 3**
     
     For example, if an image is **800x600 pixels**, the uncompressed size would be:
     
     ↳ **800 × 600 × 3 = 1,440,000 bytes** (or approximately **1,406.25 KB**).

2. **Compressed Image Size Calculation**:
   - After applying **K-means clustering** to reduce the number of colors to `k`, the image is converted to an **indexed color format**. In this format, the image contains:
     - A **color palette**: This stores up to `k` unique colors, where each color requires **3 bytes** for its RGB values.
     - **Indexed pixel data**: Each pixel is represented by an index pointing to the palette. The number of bits required for each index depends on the number of unique colors (`k`). Specifically, the number of bits per pixel is calculated as **`log2(k)`**, rounded up to the nearest whole number.
   
   - The formula for calculating the size of an indexed image is:
     
     ↳ **Size (in bytes) = (Number of Colors × 3) + (Width × Height × Bits per Pixel / 8)**
     
     For example, if `k = 16` (meaning there are 16 unique colors), and the image is **800x600 pixels**:
     - **Palette Size**: \( 16 × 3 = 48 \) bytes.
     - **Indexed Data Size**: Each pixel requires **log2(16) = 4 bits**. Therefore, the total size for indexed data is:
       
       ↳ \( 800 × 600 × 4 / 8 = 240,000 \) bytes (or **234.38 KB**).
     
     - **Total Compressed Size**: \( 48 + 240,000 = 240,048 \) bytes (or **234.38 KB**).

3. **Impact of Compression**:
   - By reducing the number of unique colors, the **indexed format** can achieve significant size reduction, especially for images with many repeated colors or gradients. This demonstrates the effectiveness of K-means clustering in compressing images without significant visual quality loss.
   - In the above example, the original image size was **1,406.25 KB**, and the compressed size was reduced to **234.38 KB**, achieving a reduction of approximately **83.33%**.

This kind of detailed size calculation and the demonstrated reduction in file size make the project highly captivating to an interviewer. It clearly showcases both the technical implementation and the practical benefits of the compression algorithm, emphasizing how well the solution can save storage space and improve data efficiency.
"""
