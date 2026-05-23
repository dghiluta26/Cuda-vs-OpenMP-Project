from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

random.seed(42)

images = [
    ("test_512x512.pgm", 512, 512),
    ("test_1024x1024.pgm", 1024, 1024),
    ("test_2048x2048.pgm", 2048, 2048),
    ("test_3840x2160.pgm", 3840, 2160),
]

def clamp(value):
    return max(0, min(255, value))

for filename, width, height in images:
    output_path = data_dir / filename

    with open(output_path, "wb") as file:
        file.write(b"P5\n")
        file.write(f"{width} {height}\n".encode())
        file.write(b"255\n")

        for y in range(height):
            for x in range(width):
                # Left side: darker region
                # Right side: brighter region
                # Small noise is added so the image is more realistic.
                if x < width // 2:
                    pixel = 50 + random.randint(-15, 15)
                else:
                    pixel = 200 + random.randint(-15, 15)

                file.write(bytes([clamp(pixel)]))

    print(f"Generated {output_path}")