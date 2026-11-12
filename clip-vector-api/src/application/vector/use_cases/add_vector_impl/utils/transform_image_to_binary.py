
from io import BytesIO

from fastapi import UploadFile

from application.vector.use_cases.add_vector_impl.exceptions.image_to_binary_exception import ImageToBinaryException


async def transform_image_to_binary(image: UploadFile) -> BytesIO:
    """Transform an image to a binary format."""
    try:
        # Read the image data
        image_data = await image.read()
        image_stream = BytesIO(image_data)
        
        if not image.content_type or not image.content_type.startswith('image/'):
            raise ImageToBinaryException(detail={"error": "File must be an image"})

        return image_stream
    except Exception as e:
        raise ImageToBinaryException(detail={"error": str(e)})