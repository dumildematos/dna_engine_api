import os
from dotenv import load_dotenv
from imagekitio import ImageKit

# Load environment variables from .env
load_dotenv()

def get_imagekit_instance():
    """
    Initializes and returns a singleton-style ImageKit instance.
    """
    return ImageKit(
        private_key=os.getenv("private_AhMbgVATSfq7+HOW0rLEhYuqPo8="),
        # public_key=os.getenv("IMAGEKIT_PUBLIC_KEY"),
        # url_endpoint=os.getenv("IMAGEKIT_URL_ENDPOINT")
    )

# Create a shared instance to be used across the app
imagekit = get_imagekit_instance()