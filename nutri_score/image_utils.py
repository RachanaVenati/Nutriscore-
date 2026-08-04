import base64
import re
from io import BytesIO

from PIL import Image


def strip_data_url_prefix(value):
    if isinstance(value, str) and "base64," in value:
        return re.sub(r"^data:.*?;base64,", "", value)
    return value


def image_from_base64(value):
    image_data = base64.b64decode(strip_data_url_prefix(value))
    return Image.open(BytesIO(image_data)).convert("RGB")

