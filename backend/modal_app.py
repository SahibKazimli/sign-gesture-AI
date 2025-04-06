import modal
from PIL import Image
import io
from transformers import pipeline

# Create a new Modal app
app = modal.App("sign-language-corrector")

# Define the container image with our dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "torch",        # For PyTorch
        "transformers", # For Hugging Face models
        "pillow"        # For image processing
    ])
)

@app.function(image=image, gpu="T4")
def process_image(image_bytes: bytes):
    """Process an image and return gesture predictions"""
    try:
        # Initialize the gesture recognition model
        gesture_classifier = pipeline(
            "image-classification",
            model="joonsong/gesture_model",
            device=0  # Use GPU
        )
        
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        results = gesture_classifier(image)
        
        # Format results
        predictions = [
            {
                "gesture": pred["label"],      # The detected gesture
                "confidence": pred["score"]     # How confident the model is (0-1)
            }
            for pred in results
        ]
        
        # Get top prediction
        top_prediction = predictions[0]
        
        # Return a detailed response
        return {
            "detected_gesture": top_prediction["gesture"],
            "confidence": top_prediction["confidence"],
            "suggestions": predictions[1:3] if len(predictions) > 1 else [],  # Alternative suggestions
            "feedback": f"Detected sign: {top_prediction['gesture']} with {top_prediction['confidence']*100:.1f}% confidence"
        }
    except Exception as e:
        return {"error": str(e)}

# Create a web endpoint that accepts POST requests
@app.function()
def process_image_endpoint(image_bytes: bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Process the image using our existing function
    result = process_image(image)
    
    return result

# For local development and testing
if __name__ == "__main__":
    modal.runner.deploy_stub(app) 