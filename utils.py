
from typing import Any


import re
import unicodedata 
from PIL import Image, UnidentifiedImageError

class Utils:
    DOUBLE_QUOTE_REGEX = re.compile(
        "|".join(["«", "‹", "»", "›", "„", "“", "‟", "”", "❝", "❞", "❮", "❯", "〝", "〞", "〟", "＂", "＂"])
    )

    SINGLE_QUOTE_REGEX = re.compile("|".join(["‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"]))

    @staticmethod
    def _remove_hieroglyphs_unicode(text: str) -> str:
        allowed_categories = {
            "Lu",  # Uppercase letter
            "Ll",  # Lowercase letter
            "Lt",  # Titlecase letter
            "Nd",  # Decimal number
            "Nl",  # Letter number
            "No",  # Other number
            "Pc",  # Connector punctuation
            "Pd",  # Dash punctuation
            "Ps",  # Open punctuation
            "Pe",  # Close punctuation
            "Pi",  # Initial punctuation
            "Pf",  # Final punctuation
            "Po",  # Other punctuation
            "Sm",  # Math symbol
            "Sc",  # Currency symbol
            "Zs",  # Space separator
        }

        result: list[str] = []

        for char in text:
            category = unicodedata.category(char)

            if category in allowed_categories:
                result.append(char)

        cleaned_text = "".join(result).strip()

        # Ensure no more than one consecutive whitespace (extra safety)
        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)

        return cleaned_text

    @staticmethod
    def _apply_punctuation_and_spacing(text: str) -> str:
        # Remove extra spaces before punctuation
        text = re.sub(r"\s+([,.!?…])", r"\1", text)

        # Ensure single space after punctuation, except for multiple punctuation marks
        text = re.sub(r"([,.!?…])(?!\s)(?![,.!?…])", r"\1 ", text)

        # Remove space between multiple punctuation marks
        text = re.sub(r"([,.!?…])\s+([,.!?…])", r"\1\2", text)

        return text.strip()

    @staticmethod
    def _fix_quotes(text: str) -> str:
        text = Utils.SINGLE_QUOTE_REGEX.sub("'", text)
        text = Utils.DOUBLE_QUOTE_REGEX.sub('"', text)
        return text

    @staticmethod
    def get_image_raw_bytes_and_dims(image_path: str) -> tuple[bytes, int, int] | None:

        try:
            with Image.open(image_path) as img:
                width = img.width
                height = img.height

            with open(image_path, 'rb') as file:
                raw_bytes = file.read()

            return (raw_bytes, width, height)

        except FileNotFoundError:
            print(f"Error: Image file not found at '{image_path}'")
            return None
        except UnidentifiedImageError:
            print(f"Error: Pillow (PIL) cannot identify '{image_path}' as an image. Cannot get dimensions.")
            return None
        except IOError as e:
            print(f"Error reading raw file bytes from '{image_path}': {e}")
            return None
        except Exception as e:
            # Add type hints to help static analysis if possible, but Exception is broad
            print(f"An unexpected error occurred processing '{image_path}': {e}")
            return None