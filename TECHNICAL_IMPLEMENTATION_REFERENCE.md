# X-Ray Validation Implementation - Technical Reference

## Architecture Overview

```
User Uploads Image
    ↓
[Frontend: index.html + app.js]
    - File type validation (.png, .jpg, .jpeg)
    - File size check (max 16MB)
    - Display preview
    ↓
POST /predict API
    ↓
[Backend: app.py]
    └─→ predict_image(image_bytes)
        ├─→ Load image as grayscale
        │
        ├─→ validate_chest_xray(image)
        │   ├─ Check 1: Dimensions
        │   ├─ Check 2: Aspect Ratio
        │   ├─ Check 3: Intensity Stats
        │   ├─ Check 4: Histogram
        │   ├─ Check 5: Sobel Edges
        │   ├─ Check 6: Laplacian HF
        │   └─ Check 7: Quantiles
        │   └─ Returns: True/False
        │
        ├─ If False → Return error message
        │
        ├─ If True → Preprocess image
        ├─ Run model prediction
        ├─ Check confidence >= 0.55
        ├─ If confidence too low → Return error message
        └─ If valid → Return prediction results
    ↓
Return JSON Response
    ↓
[Frontend: Display results or error]
```

## Code Implementation Details

### 1. Import Statement (app.py, line 12)
```python
from scipy import ndimage
```
**Used for**: Sobel edge detection and Laplacian filtering

### 2. validate_chest_xray() Function
**Location**: app.py, lines 135-231  
**Returns**: Boolean (True if valid X-ray, False otherwise)

#### Check 1: Image Dimensions
```python
if height < 50 or width < 50:
    return False
```
**Logic**: X-rays must have minimum dimensions to show anatomical structures

#### Check 2: Aspect Ratio
```python
aspect_ratio = width / height if height > 0 else 0
if aspect_ratio > 1.5 or aspect_ratio < 0.3:
    return False
```
**Logic**: Chest X-rays are portrait or nearly square, not extremely wide/tall

#### Check 3: Intensity Statistics
```python
mean_intensity = np.mean(img_gray)
std_intensity = np.std(img_gray)

if mean_intensity < 20 or mean_intensity > 230:
    return False
if std_intensity < 12:
    return False
```
**Logic**: 
- Mean range (20-230): Avoids too-dark or over-exposed images
- Std (>12): Ensures sufficient contrast (rules out solid colors)

#### Check 4: Histogram Distribution
```python
histogram, _ = np.histogram(img_gray, bins=256, range=(0, 256))

peaks = 0
for i in range(1, 255):
    if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
        peaks += 1

if peaks < 2:
    return False
```
**Logic**: X-rays have bimodal/multi-modal histograms due to different tissue densities

#### Check 5: Sobel Edge Detection
```python
sx = ndimage.sobel(img_gray, axis=0)  # Vertical edges
sy = ndimage.sobel(img_gray, axis=1)  # Horizontal edges
edges = np.sqrt(sx**2 + sy**2)        # Edge magnitude
edge_mean = np.mean(edges)

if edge_mean < 5 or edge_mean > 150:
    return False
```
**Logic**: X-rays have characteristic edge patterns from bones and anatomical structures
- Too low: Uniform photos
- Too high: Noisy/random images

#### Check 6: Laplacian High-Frequency Content
```python
laplacian = ndimage.laplace(img_gray)
high_freq = np.std(laplacian)

if high_freq < 1:
    return False
```
**Logic**: X-rays have moderate detail (high-frequency content)
- Too low: Blurry or featureless images

#### Check 7: Quantile Analysis
```python
p25 = np.percentile(img_gray, 25)
p75 = np.percentile(img_gray, 75)
percentile_range = p75 - p25

if percentile_range < 30:
    return False
```
**Logic**: X-rays use diverse intensity values
- Measures interquartile range (IQR)
- Solid colors have IQR ≈ 0

### 3. predict_image() Function
**Location**: app.py, lines 233-283  
**Returns**: Dictionary with success/error/prediction

#### Step 1: Load and Validate
```python
image = Image.open(io.BytesIO(image_bytes)).convert('L')
is_valid_xray = validate_chest_xray(image)

if not is_valid_xray:
    return {
        'success': False,
        'error': 'Error: Please upload a valid chest or lung X-ray image only.'
    }
```

#### Step 2: Preprocess
```python
transform = get_transform()
image_tensor = transform(image).unsqueeze(0).to(device)
```

#### Step 3: Predict
```python
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
```

#### Step 4: Confidence Check
```python
CONFIDENCE_THRESHOLD = 0.55

if confidence < CONFIDENCE_THRESHOLD:
    return {
        'success': False,
        'error': 'Error: Please upload a valid chest or lung X-ray image only.'
    }
```
**Logic**: Secondary validation to catch edge cases where validation passes but model is uncertain

#### Step 5: Return Results
```python
return {
    'success': True,
    'prediction': Config.CLASSES[predicted_class],
    'class_index': predicted_class,
    'normal_confidence': float(probs[0]),
    'pneumonia_confidence': float(probs[1]),
    'confidence': confidence
}
```

### 4. Error Handling

**Unified Error Message**:
```python
'Error: Please upload a valid chest or lung X-ray image only.'
```

**Triggered by**:
1. Failed image validation (Check 1-7)
2. Low model confidence
3. Any exception during processing

## Performance Characteristics

### Computational Complexity
- Histogram generation: O(n) where n = image pixels
- Sobel filtering: O(n) convolution
- Laplacian filtering: O(n) convolution
- Quantile calculation: O(n log n) sorting
- **Total**: ~50-100ms per image (depending on resolution)

### Memory Usage
- Grayscale image: height × width × 1 byte
- Intermediate arrays (same size)
- **Example**: 224×224 image ≈ 50KB working memory

## Threshold Tuning

### Conservative (Stricter - More False Rejections)
```python
# Validation thresholds
peaks < 3  # Require more histogram peaks
std_intensity < 15  # Higher contrast requirement
edge_mean: 8 to 120  # Narrower edge range
CONFIDENCE_THRESHOLD = 0.65  # Higher model confidence
```

### Permissive (Looser - More False Positives)
```python
# Validation thresholds
peaks < 1  # Allow single peak
std_intensity < 8  # Lower contrast requirement
edge_mean: 2 to 200  # Wider edge range
CONFIDENCE_THRESHOLD = 0.45  # Lower model confidence
```

### Recommended (Current - Balanced)
```python
# Current settings in code
peaks < 2
std_intensity < 12
edge_mean: 5 to 150
CONFIDENCE_THRESHOLD = 0.55
```

## Dependencies

### Required for Implementation
- **numpy**: Array operations, statistics
- **scipy.ndimage**: Sobel, Laplacian filters
- **PIL**: Image loading and conversion
- **torch**: Model inference

### Added to requirements.txt
```
scipy>=1.7.0
```

## Testing Strategies

### Unit Testing Approach
```python
def test_validate_xray():
    # Valid X-rays should pass
    valid_xray = Image.open('sample_xray.png')
    assert validate_chest_xray(valid_xray) == True
    
    # Non-medical images should fail
    photo = Image.open('photo.jpg')
    assert validate_chest_xray(photo) == False
```

### Integration Testing
- Test full prediction pipeline with various image types
- Verify error messages display correctly
- Check API response format

### Statistical Testing
- Measure false positive rate (non-X-rays passing validation)
- Measure false negative rate (valid X-rays rejected)
- Goal: <5% false positive, <2% false negative

## Future Enhancements

1. **ML-based Validation**
   - Train binary classifier (X-ray vs non-X-ray)
   - More robust than heuristic rules
   
2. **Image Logging**
   - Log rejected images for analysis
   - Identify edge cases for threshold adjustment
   
3. **Region-of-Interest Detection**
   - Detect chest boundaries
   - Validate lungs visible
   
4. **Anatomical Landmark Detection**
   - Use computer vision to find characteristic features
   - Detect heart, ribs, spine
   
5. **Adaptive Thresholds**
   - Adjust validation thresholds based on image type
   - Different settings for PA vs lateral views

---

**Version**: 1.0  
**Last Updated**: 2026-04-30  
**Maintainer**: Development Team
