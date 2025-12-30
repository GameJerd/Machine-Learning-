import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image, ImageOps

# --- PHASE 1: TRAIN THE AI (The "Knowledge Base") ---

print("Training the model on standard digits...")

# 1. Load Data
digits = load_digits()
X = digits.data
y = digits.target

# 2. Simulate Real-World Messiness (Corrupt the data)
# We delete 15% of pixels to force the model to learn 'robustness'
rng = np.random.RandomState(42)
X_corrupted = X.copy()
mask = rng.rand(*X.shape) < 0.15
X_corrupted[mask] = np.nan

# 3. Build the Processing Pipeline (Must keep these objects to use later!)
# Step A: Imputer (Fix missing holes)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_corrupted)

# Step B: Standardization (Center data)
scaler = StandardScaler()
X_std = scaler.fit_transform(X_imputed)

# Step C: PCA (Compress to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

print("Model trained! The AI can now map 64-pixel images to a 2D map.")


# --- PHASE 2: THE INTERACTIVE FUNCTION ---

def test_my_image(image_path):
    """
    Reads a local image file, converts it to 8x8 format,
    runs it through the AI pipeline, and plots it.
    """
    try:
        # 1. Load Image using PIL
        img = Image.open(image_path).convert('L')  # Convert to Grayscale

        # 2. Invert colors (If you drew black digit on white paper)
        # The dataset expects White Digit on Black Background.
        # We assume the user drew on a white background, so we invert.
        img = ImageOps.invert(img)

        # 3. Resize to 8x8 pixels (The required resolution)
        img = img.resize((8, 8), Image.Resampling.LANCZOS)

        # 4. Convert to NumPy array and Scale to 0-16 range
        img_array = np.array(img)
        # Normal images are 0-255. We scale to 0-16.
        img_array = (img_array / 255.0) * 16.0

        # 5. Flatten to (1, 64) shape
        # The model expects a flat row of numbers, not a grid.
        feature_vector = img_array.reshape(1, -1)

        # --- RUN THE PIPELINE ---
        # Note: We use .transform(), NOT .fit_transform()
        # We must use the *existing* logic the AI learned earlier.
        
        # A. Impute (even if no missing values, we must pass through it)
        feat_imputed = imputer.transform(feature_vector)
        
        # B. Standardize (using the mean/std learned from the original data)
        feat_std = scaler.transform(feat_imputed)
        
        # C. PCA (Project into the 2D map)
        feat_pca = pca.transform(feat_std)

        # --- PLOT RESULTS ---
        plt.figure(figsize=(12, 6))

        # Subplot 1: The map
        plt.subplot(1, 2, 1)
        # Plot the original dataset as background
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.3, s=20)
        # Plot the USER'S new point
        plt.scatter(feat_pca[0, 0], feat_pca[0, 1], c='red', s=200, marker='*', label='Your Image')
        
        plt.legend()
        plt.title(f"Where your digit landed\n(Coords: {feat_pca[0,0]:.2f}, {feat_pca[0,1]:.2f})")
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True, alpha=0.3)

        # Subplot 2: What the AI actually "saw"
        plt.subplot(1, 2, 2)
        plt.imshow(img_array, cmap='gray') # Show the 8x8 grid
        plt.title("What the AI saw (8x8 Input)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
        print("Success! Check the plot to see which cluster your digit landed in.")

    except FileNotFoundError:
        print(f"Error: Could not find file '{image_path}'. Please check the name.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- PHASE 3: RUN IT ---

# INSTRUCTIONS FOR YOU:
# 1. Open MS Paint or Photoshop.
# 2. Make a square canvas (e.g., 200x200).
# 3. Draw a big black number (0-9) on the white background.
# 4. Save it as "my_digit.png" in the same folder as this script.
# 5. Run the line below:

# Uncomment this line when you have a file ready!
# test_my_image("my_digit.png")

# --- DEMO MODE ---
# Since I cannot see your files, I will generate a fake "Handwritten 0"
# to demonstrate what happens when you run the function.
print("\n--- Running Demo Mode (Simulating a user uploading a '0') ---")
dummy_zero = np.zeros((8, 8))
dummy_zero[1:7, 2:6] = 16  # Draw a box (looks like a 0)
dummy_zero[2:6, 3:5] = 0   # Hollow out the center

# We save this as a temp file just to test our function
from PIL import Image
# Invert for saving (because our function expects black-on-white input to invert back)
demo_img = Image.fromarray((255 - (dummy_zero * (255/16))).astype('uint8')).convert('L')
demo_img.save("demo_zero.png")

test_my_image("demo_zero.png")