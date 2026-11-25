# Regenerates installer/msix/Assets and installer/eullm.ico from the logo:
#   python3 installer/msix/make-icons.py installer/msix/Assets
# Needs Pillow (a local tool, not a build dependency: the images are committed).
# The icon is the logo's mark (the "E" ribbons), cut out of
# branding/eullm-logo-source.png as its first run of non-empty columns, so a
# new logo there is all it takes to redraw every icon.
import os
import sys

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE = os.path.join(ROOT, "branding", "eullm-logo-source.png")
WHITE = (255, 255, 255, 255)


def mark(logo):
    """The logo's mark: its leftmost run of columns that hold any ink."""
    alpha = logo.split()[3]
    ink = [any(alpha.getpixel((x, y)) > 20 for y in range(logo.height)) for x in range(logo.width)]
    start = ink.index(True)
    end = start
    while end < len(ink) and ink[end]:
        end += 1
    cut = logo.crop((start, 0, end, logo.height))
    return cut.crop(cut.getbbox())


def square(img, size, fill, bg):
    """`img` centred on a size x size canvas, its longer side `fill` of it."""
    canvas = Image.new("RGBA", (size, size), bg)
    scale = fill * size / max(img.size)
    small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                       Image.LANCZOS)
    canvas.alpha_composite(small, ((size - small.width) // 2, (size - small.height) // 2))
    return canvas


def main(out):
    os.makedirs(out, exist_ok=True)
    logo = Image.open(SOURCE).convert("RGBA")
    logo = logo.crop(logo.getbbox())
    m = mark(logo)

    def save(img, name):
        img.save(os.path.join(out, name), optimize=True)

    # Plated tiles: the mark on white, as the logo itself is drawn.
    for s in (44, 150):
        save(square(m, s, 0.72, WHITE), f"Square{s}x{s}Logo.png")
    save(square(m, 50, 0.72, WHITE), "StoreLogo.png")
    # Taskbar / terminal sizes without the plate, as Windows expects for unplated targets.
    for s in (16, 24, 32, 48, 256):
        save(square(m, s, 0.92, (0, 0, 0, 0)),
             f"Square44x44Logo.targetsize-{s}_altform-unplated.png")
    # The wide tile carries the whole logo.
    wide = Image.new("RGBA", (310, 150), WHITE)
    scale = min(280 / logo.width, 120 / logo.height)
    small = logo.resize((round(logo.width * scale), round(logo.height * scale)), Image.LANCZOS)
    wide.alpha_composite(small, ((310 - small.width) // 2, (150 - small.height) // 2))
    save(wide, "Wide310x150Logo.png")
    # The Inno Setup installer's icon (installer/eullm.iss: SetupIconFile, shortcuts).
    square(m, 256, 0.92, (0, 0, 0, 0)).save(
        os.path.join(ROOT, "installer", "eullm.ico"),
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


if __name__ == "__main__":
    main(sys.argv[1])
