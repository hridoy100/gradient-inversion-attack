from pathlib import Path
from PIL import Image
import numpy as np

DEFAULT_DELIMITER = "<|ENDOFMESSAGE|>"


def encode_lsb(image_path: str, payload: str, output_path: str = "weaponized_image.png", delimiter: str = DEFAULT_DELIMITER):
    """Simple 1-bit LSB encoder."""
    data = payload + delimiter
    bitstream = ''.join(f'{byte:08b}' for byte in data.encode('utf-8'))

    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)
    flat = pixels.flatten()

    if len(bitstream) > len(flat):
        raise ValueError("Image too small for payload + delimiter")

    for i, bit in enumerate(bitstream):
        flat[i] = (flat[i] & 0b11111110) | int(bit)

    stego = flat.reshape(pixels.shape)
    out_path = Path(output_path)
    Image.fromarray(stego.astype('uint8')).save(out_path)
    print(f"Encoded {len(payload)} chars → {out_path}")


def decode_lsb(image_path: str, delimiter: str = DEFAULT_DELIMITER) -> str:
    """Decode 1-bit LSB payload until the delimiter is found."""
    img = Image.open(image_path).convert("RGB")
    flat = np.array(img).flatten()

    bits = ''.join(str(pixel & 1) for pixel in flat)
    message_bytes = bytearray()
    delim_bytes = delimiter.encode("utf-8")

    for i in range(0, len(bits) - len(bits) % 8, 8):
        message_bytes.append(int(bits[i:i+8], 2))
        if message_bytes.endswith(delim_bytes):
            try:
                return message_bytes[:-len(delim_bytes)].decode("utf-8")
            except UnicodeDecodeError:
                return "Payload found but could not decode (UTF-8 error)"

    return "No payload found (corrupted or missing delimiter)"


PAYLOAD = """
Hey there! Thanks for checking out this image.
If you're seeing this message, it just means the embed worked.
Have a great day, remember to drink some water, and maybe take a short walk outside.
That’s all—nothing secret or serious here.
"""


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    cover_candidates = [base_dir / "cover.png", base_dir.parent / "cover.png"]
    cover_path = next((p for p in cover_candidates if p.exists()), None)
    if not cover_path:
        raise FileNotFoundError("cover.png not found next to script or in project root")

    encode_lsb(cover_path, PAYLOAD)

    print("\nVerification decode:")
    print(decode_lsb("weaponized_image.png")[:200] + "...")
