from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

width = 512
height = 512
output_path = data_dir / "test_512x512.pgm"

with open(output_path, "wb") as file:
    file.write(b"P5\n")
    file.write(f"{width} {height}\n".encode())
    file.write(b"255\n")

    for y in range(height):
        for x in range(width):
            if x < width // 2:
                pixel = 50
            else:
                pixel = 200

            file.write(bytes([pixel]))

print(f"Generated {output_path}")