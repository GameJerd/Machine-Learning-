import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

from PIL import Image, ImageOps

# STEP 2: LOAD THE TRAINING DATA

digits = load_digits()
X = digits.data
y = digits.target

print(f"Step 1: Loaded {len(X)} digit images (8x8 pixels each)")


np.random.seed(42)
X_messy = X.copy()
missing_mask = np.random.rand(*X.shape) < 0.15
X_messy[missing_mask] = np.nan

print(f"Step 2: Simulated messy data (15% of pixels missing)")


imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X_messy)

print(f"Step 3: Imputation complete (filled missing values with median)")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

print(f"Step 4: Standardization complete (all features now on same scale)")

# STEP 6: PERSONALIZED LEARNING - Add Your Handwriting Style

extra_images = []
extra_labels = []

if os.path.exists("my_digit.png"):
    try:
        img = Image.open("my_digit.png")
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        img = ImageOps.invert(img.convert('L'))
        img_array = np.array(img)
        
        pixels = np.argwhere(img_array > 30)
        if len(pixels) > 0:
            top, left = pixels.min(axis=0)
            bottom, right = pixels.max(axis=0)
            padding = int(max(bottom-top, right-left) * 0.15)
            top, left = max(0, top-padding), max(0, left-padding)
            bottom = min(img_array.shape[0], bottom+padding)
            right = min(img_array.shape[1], right+padding)
            
            cropped = img_array[top:bottom, left:right]
            size = max(cropped.shape)
            square = np.zeros((size, size), dtype=np.uint8)
            h, w = cropped.shape
            square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = cropped
            small = np.array(Image.fromarray(square).resize((8, 8), Image.Resampling.LANCZOS))
            your_digit = small / 255.0 * 16.0
            
            for _ in range(30):
                variation = your_digit.flatten() + np.random.normal(0, 0.5, 64)
                extra_images.append(np.clip(variation, 0, 16))
                extra_labels.append(7)
            
            print(f"Step 5: Added 30 samples of YOUR handwriting")
    except:
        pass

if extra_images:
    X_combined = np.vstack([X_clean, np.array(extra_images)])
    y_combined = np.concatenate([y, np.array(extra_labels)])
    X_scaled = scaler.fit_transform(X_combined)
    y = y_combined
else:
    print(f"Step 5: No custom image found (save 'my_digit.png' to personalize)")

# STEP 7: PCA - Reducing Dimensions for Visualization

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

print(f"Step 6: PCA complete (64 dimensions → 2 for visualization)")

# STEP 8: TRAIN THE CLASSIFIER

knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
svm = SVC(kernel='rbf', probability=True, C=10)

classifier = VotingClassifier(
    estimators=[('knn', knn), ('svm', svm)],
    voting='soft'
)
classifier.fit(X_scaled, y)

print(f"Step 7: Classifier trained (KNN + SVM ensemble)")

# THE PREDICTION FUNCTION

def predict_digit(image_path):
    """Reads image file, preprocesses it, and predicts the digit"""
    
    print("ANALYZING YOUR IMAGE....")
    
    try:
        img = Image.open(image_path)
        print(f"Loaded: {image_path}")
        
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        
        img = ImageOps.invert(img.convert('L'))
        img_array = np.array(img)
        
        pixels = np.argwhere(img_array > 30)
        if len(pixels) == 0:
            print("Error: No digit found in image!")
            return
        
        top, left = pixels.min(axis=0)
        bottom, right = pixels.max(axis=0)
        padding = int(max(bottom-top, right-left) * 0.15)
        top, left = max(0, top-padding), max(0, left-padding)
        bottom = min(img_array.shape[0], bottom+padding)
        right = min(img_array.shape[1], right+padding)
        
        cropped = img_array[top:bottom, left:right]
        
        size = max(cropped.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        h, w = cropped.shape
        square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = cropped
        small = Image.fromarray(square).resize((8, 8), Image.Resampling.LANCZOS)
        digit_8x8 = np.array(small, dtype=np.float64) / 255.0 * 16.0
        
        print(f"Processed: Cropped, squared, resized to 8x8")
        
        features = digit_8x8.reshape(1, -1)
        
        features = imputer.transform(features)
        features = scaler.transform(features)
        
        prediction = classifier.predict(features)[0]
        probabilities = classifier.predict_proba(features)[0]
        confidence = probabilities[prediction] * 100
        
        top3 = np.argsort(probabilities)[::-1][:3]
        
        print(f"PREDICTION: The digit is  [ {prediction} ]")
        print(f"CONFIDENCE: {confidence:.1f}%")
        print("Top 3 guesses:")
        for i, digit in enumerate(top3):
            marker = " <<<" if i == 0 else ""
            print(f"   {i+1}. Digit {digit}: {probabilities[digit]*100:.1f}%{marker}")
        
        features_2d = pca.transform(features)
        
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f"Prediction: Digit {prediction} ({confidence:.1f}% confidence)", 
                    fontsize=14, fontweight='bold')
        
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.3, s=15)
        plt.scatter(features_2d[0, 0], features_2d[0, 1], c='red', s=200, marker='*', 
                   label='Your digit', edgecolors='black')
        plt.colorbar(scatter, label='Digit')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Map\n(Your digit = red star)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.imshow(digit_8x8, cmap='gray', vmin=0, vmax=16)
        plt.title('What AI Sees\n(Your image as 8x8 pixels)')
        plt.axis('off')
        for i in range(8):
            for j in range(8):
                val = digit_8x8[i, j]
                if val > 1:
                    color = 'black' if val > 8 else 'white'
                    plt.text(j, i, f'{val:.0f}', ha='center', va='center', 
                            fontsize=6, color=color)
        
        plt.subplot(1, 3, 3)
        examples = np.zeros((20, 45))
        for d in range(10):
            idx = np.where(digits.target == d)[0][0]
            row, col = d // 5, d % 5
            examples[row*10:row*10+8, col*9:col*9+8] = digits.images[idx]
        plt.imshow(examples, cmap='gray')
        plt.title('Training Examples (0-9)\n(Compare with your digit)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return prediction
        
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found!")
        print("Save your digit image as 'my_digit.png' in this folder.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\nTesting with your image...")
    predict_digit("my_digit.png")