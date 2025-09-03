#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCIO-only S-Log3 → ACES → Display JPEG converter

Defaults:
  - Input images from ./input/
  - Output JPEGs to ./output/
  - OCIO configs from ./ACES_Ver/

User can override with:
  --input-dir ./myphotos
  --output-dir ./results
  --aces-dir ./configs

Workflow:
  - Pick OCIO config
  - Pick input colorspace
  - Pick display + view
  - Enter EV (exposure compensation)
  - Convert all images and save JPEGs with date + parameters in filename
"""

from __future__ import annotations
from pathlib import Path
import sys, datetime, argparse
import cupy as np
from PIL import Image
import pillow_heif

try:
    import PyOpenColorIO as ocio
except Exception:
    print("PyOpenColorIO is required. Install with: pip install PyOpenColorIO", file=sys.stderr)
    raise

pillow_heif.register_heif_opener()


# ---------- I/O ----------
def read_image_as_float(path: Path) -> tuple[np.ndarray, dict]:
    img = Image.open(path)
    if img.mode not in ("RGB", "I;16", "I;16B", "I"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
    else:
        x = arr.astype(np.float32)
        m = float(x.max()) if x.size else 1.0
        if m > 0:
            x /= m
    return x, (getattr(img, "info", {}) or {})


def save_jpeg(path: Path, rgb_8bit: np.ndarray, quality: int = 100, exif: bytes | None = None):
    img = Image.fromarray(rgb_8bit, mode="RGB")
    kwargs = {"quality": int(quality), "subsampling": "4:4:4", "optimize": True}
    if exif:
        try:
            if len(exif) <= 65533:
                kwargs["exif"] = exif
                img.save(path, format="JPEG", **kwargs)
                return
            try:
                import piexif
                d = piexif.load(exif)
                d["Exif"].pop(piexif.ExifIFD.MakerNote, None)
                d["thumbnail"] = None
                for ifd in ("0th", "1st"):
                    for tag in list(d.get(ifd, {}).keys()):
                        if tag in (0x9C9B, 0x9C9C, 0x9C9D, 0x9C9E, 0x9C9F):
                            d[ifd].pop(tag, None)
                exif2 = piexif.dump(d)
                if len(exif2) <= 65533:
                    kwargs["exif"] = exif2
            except Exception:
                pass
        except Exception:
            pass
    img.save(path, format="JPEG", **kwargs)


# ---------- OCIO ----------
def list_ocio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = list(root.glob("*.ocio")) + list(root.glob("**/*.ocio"))
    uniq, seen = [], set()
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)
    uniq.sort()
    return uniq


def choose_from_list(title: str, items: list[str]) -> int:
    print(f"\n== {title} ==")
    for i, it in enumerate(items):
        print(f"[{i}] {it}")
    while True:
        s = input("Enter index: ").strip()
        try:
            idx = int(s)
            if 0 <= idx < len(items):
                return idx
        except Exception:
            pass
        print("Invalid index. Try again.")


def sorted_input_colorspaces(cfg: "ocio.Config") -> list[str]:
    names = [cs.getName() for cs in cfg.getColorSpaces()]
    prio, rest = [], []
    for n in names:
        ln = n.lower()
        if "slog3" in ln or "s-log3" in ln:
            prio.append(n)
        else:
            rest.append(n)
    return prio + rest


def ocio_apply_pipeline(
    img_enc_rgb: np.ndarray,
    cfg: "ocio.Config",
    input_cs: str,
    display: str,
    view: str,
    ev: float = 0.0,
) -> np.ndarray:
    h, w, _ = img_enc_rgb.shape
    buf = img_enc_rgb.astype(np.float32, copy=True).reshape(-1, 3)

    ap0_name = "ACES2065-1"
    if not any(cs.getName() == ap0_name for cs in cfg.getColorSpaces()):
        ap0_name = "ACES - ACES2065-1"

    proc_idt = cfg.getProcessor(input_cs, ap0_name).getDefaultCPUProcessor()
    proc_idt.applyRGB(buf)
    ap0 = buf.reshape(h, w, 3)

    if ev != 0.0:
        ap0 *= (2.0 ** ev)

    dvt = ocio.DisplayViewTransform(src=ap0_name, display=display, view=view)
    proc_dvt = cfg.getProcessor(dvt).getDefaultCPUProcessor()
    buf2 = ap0.reshape(-1, 3)
    proc_dvt.applyRGB(buf2)
    disp = buf2.reshape(h, w, 3)

    return np.clip(disp, 0.0, 1.0)


# ---------- Main ----------
def convert_one(
    input_path: Path,
    output_dir: Path,
    cfg: "ocio.Config",
    cfg_path: Path,
    input_cs: str,
    display: str,
    view: str,
    ev: float = 0.0,
    quality: int = 100,
    keep_exif: bool = False,
):
    rgb_log, info = read_image_as_float(input_path)

    disp = ocio_apply_pipeline(rgb_log, cfg, input_cs, display, view, ev=ev)
    out8 = np.clip(np.round(disp * 255.0), 0, 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    safe_in = input_cs.replace(" ", "_").replace("/", "-")
    safe_disp = display.replace(" ", "_")
    safe_view = view.replace(" ", "_")

    out_name = f"{input_path.stem}_{today}_{safe_in}_{safe_disp}_{safe_view}_EV{ev:+.2f}.jpg"
    out_path = output_dir / out_name

    exif_bytes = info.get("exif") if keep_exif else None
    save_jpeg(out_path, out8, quality=quality, exif=exif_bytes)

    print(f"✓ Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="OCIO-only S-Log3 to ACES JPEG converter")
    ap.add_argument("--input-dir", type=Path, default=Path("./Input"), help="Input folder (default ./input)")
    ap.add_argument("--output-dir", type=Path, default=Path("./Output"), help="Output folder (default ./output)")
    ap.add_argument("--aces-dir", type=Path, default=Path("./ACES_Ver"), help="ACES config folder (default ./ACES_Ver)")
    ap.add_argument("--recursive", action="store_true", help="Recurse input subfolders for images")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality (default 100)")
    ap.add_argument("--keep-exif", action="store_true", help="Try to keep EXIF (oversize trimmed or dropped)")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    aces_root = args.aces_dir.resolve()

    if not input_dir.exists():
        print(f"No input folder found at {input_dir}")
        sys.exit(1)

    ocio_files = list_ocio_files(aces_root)
    if not ocio_files:
        print(f"No .ocio files found under {aces_root}")
        sys.exit(2)

    idx_cfg = choose_from_list("Pick an ACES/OCIO config", [str(p) for p in ocio_files])
    cfg_path = ocio_files[idx_cfg]
    print(f"\nUsing OCIO config: {cfg_path}")
    cfg = ocio.Config.CreateFromFile(str(cfg_path))
    ocio.SetCurrentConfig(cfg)

    inputs = sorted_input_colorspaces(cfg)
    idx_in = choose_from_list("Pick INPUT colorspace", inputs)
    input_cs = inputs[idx_in]

    displays = list(cfg.getDisplays())
    idx_disp = choose_from_list("Pick DISPLAY", displays)
    display = displays[idx_disp]

    views = list(cfg.getViews(display))
    idx_view = choose_from_list(f"Pick VIEW for {display}", views)
    view = views[idx_view]

    s = input("\nExposure compensation EV (default 0.0). Enter value or press Enter: ").strip()
    ev = 0.0
    if s:
        try:
            ev = float(s)
        except Exception:
            print("Invalid EV, using 0.0")

    exts = ("*.HIF","*.HEIF","*.HEIC","*.JPG","*.JPEG","*.PNG")
    if args.recursive:
        files = []
        for ext in exts:
            files.extend(input_dir.glob(ext))
        files = sorted(files)
    else:
        files = []
        for ext in exts:
            files.extend(input_dir.glob(ext))
        files = sorted(files)

    if not files:
        print(f"No images found in {input_dir}")
        sys.exit(3)

    for f in files:
        convert_one(
            f, output_dir, cfg, cfg_path, input_cs, display, view,
            ev=ev, quality=args.quality, keep_exif=args.keep_exif
        )


if __name__ == "__main__":
    main()
