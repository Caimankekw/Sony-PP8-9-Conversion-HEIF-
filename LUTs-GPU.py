#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slog3_heif_to_jpeg_lut.py (CuPy accelerated)
--------------------------------------------
Batch convert Sony HIF/HEIF (PP8 S-Log3 + S-Gamut3.Cine) to sRGB JPEG via a selected .cube 3D LUT.

GPU policy:
- All math runs on GPU via CuPy when available.
- Pillow/pillow-heif stay on CPU; we convert arrays at the edges via to_gpu()/to_cpu().
"""

from __future__ import annotations
from pathlib import Path
import sys, datetime, argparse

# ---------- CPU/GPU unification ----------
try:
    import cupy as cp
    _GPU = True
except Exception:
    cp = None
    _GPU = False

import numpy as _np  # always-available NumPy for I/O (Pillow, file parsing)
xp = cp if _GPU else _np

from PIL import Image
try:
    import pillow_heif
except Exception:
    pillow_heif = None

# ---------- bridge helpers ----------
def to_gpu(a):
    """Move array to GPU (CuPy) if available; else return as-is."""
    if _GPU:
        try:
            if isinstance(a, cp.ndarray):
                return a
            return cp.asarray(a)
        except Exception:
            return a
    return a

def to_cpu(a):
    """Move array to CPU (NumPy). Robust to cp.ndarray and cuda-aware arrays."""
    if _GPU:
        try:
            if isinstance(a, cp.ndarray):
                return cp.asnumpy(a)
        except Exception:
            pass
        # Objects with __cuda_array_interface__ but not cp.ndarray
        try:
            return _np.array(a)  # best-effort fallback
        except Exception:
            return a
    return a

# -------------------------------
# S-Log3 <-> linear (Sony PP8)
# -------------------------------
def _slog3_break_and_intercept():
    C1 = 0.01125000
    C2 = 0.037584
    A  = 0.432699
    B  = 0.616596
    E  = 0.03
    F  = 3.53881278538813
    # compute with CPU float to avoid dtype surprises
    S_break = A * _np.log10(C1 + C2) + B + E
    intercept = S_break - F * C1
    return C1, C2, A, B, E, F, float(S_break), float(intercept)

_C1, _C2, _A, _B, _E, _F, _S_BREAK, _INTERCEPT = _slog3_break_and_intercept()

def slog3_to_linear(x):
    x = x.astype(xp.float32)
    high = x >= _S_BREAK
    y = xp.empty_like(x, dtype=xp.float32)
    y[high]  = xp.power(10.0, (x[high] - _B - _E) / _A) - _C2
    y[~high] = (x[~high] - _INTERCEPT) / _F
    return y

def linear_to_slog3(y):
    y = y.astype(xp.float32)
    high = y >= _C1
    x = xp.empty_like(y, dtype=xp.float32)
    x[high]  = _A * xp.log10(xp.maximum(y[high] + _C2, 1e-10)) + _B + _E
    x[~high] = _F * y[~high] + _INTERCEPT
    return x

# -------------------------------
# sRGB OETF
# -------------------------------
def srgb_oetf(linear):
    a = 0.055
    linear = xp.clip(linear.astype(xp.float32), 0.0, 1.0)
    low = linear <= 0.0031308
    out = xp.empty_like(linear, dtype=xp.float32)
    out[low]  = 12.92 * linear[low]
    out[~low] = (1 + a) * xp.power(linear[~low], 1/2.4) - a
    return out

# -------------------------------
# .cube LUT (3D) parsing + tri-linear
# -------------------------------
class CubeLUT3D:
    def __init__(self, size: int, table, domain_min=None, domain_max=None):
        self.size = int(size)
        # table expected on GPU for fast sampling
        self.table = to_gpu(table).astype(xp.float32, copy=False)
        if self.table.shape != (self.size, self.size, self.size, 3):
            raise ValueError("LUT table shape mismatch.")
        self.domain_min = to_gpu(_np.array([0.0, 0.0, 0.0], dtype=_np.float32) if domain_min is None else _np.asarray(domain_min, dtype=_np.float32))
        self.domain_max = to_gpu(_np.array([1.0, 1.0, 1.0], dtype=_np.float32) if domain_max is None else _np.asarray(domain_max, dtype=_np.float32))

    @staticmethod
    def from_cube_file(path: Path) -> "CubeLUT3D":
        size = None; domain_min = None; domain_max = None; data = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                toks = s.split()
                key = toks[0].upper()
                if key == 'TITLE':
                    continue
                elif key == 'LUT_3D_SIZE':
                    size = int(toks[1])
                elif key == 'DOMAIN_MIN':
                    domain_min = _np.array(list(map(float, toks[1:4])), dtype=_np.float32)
                elif key == 'DOMAIN_MAX':
                    domain_max = _np.array(list(map(float, toks[1:4])), dtype=_np.float32)
                elif key.startswith('LUT_1D_SIZE') or key.startswith('LUT_2D_SIZE'):
                    raise ValueError("Only 3D LUTs are supported.")
                else:
                    if len(toks) >= 3:
                        try:
                            r, g, b = float(toks[0]), float(toks[1]), float(toks[2])
                            data.append((r, g, b))
                        except Exception:
                            continue
        if size is None:
            raise ValueError("No LUT_3D_SIZE found in .cube file.")
        expected = size * size * size
        if len(data) != expected:
            raise ValueError(f".cube data length {len(data)} != expected {expected}")
        # build on CPU, then move to GPU
        table_np = _np.array(data, dtype=_np.float32).reshape((size, size, size, 3))
        return CubeLUT3D(size=size, table=table_np, domain_min=domain_min, domain_max=domain_max)

    def to_axes_rgb_inplace(self, scan_order: str = 'r_g_b'):
        """
        Canonicalize axes to [R,G,B] so we can index table[r,g,b].
        scan_order:
          - 'r_g_b' (DEFAULT/common): raw is (B,G,R,3) -> transpose to (R,G,B,3).
          - 'b_g_r': raw already (R,G,B,3).
        """
        if scan_order == 'r_g_b':
            self.table = xp.transpose(self.table, (2,1,0,3))
        elif scan_order == 'b_g_r':
            pass
        else:
            raise ValueError("scan_order must be 'r_g_b' or 'b_g_r'.")

    def apply(self, img):
        """
        Tri-linear sample on GPU.
        Assumes self.table is [R,G,B,3] and input img is float32 RGB in [0,1] (GPU array).
        """
        img = img.astype(xp.float32, copy=False)
        H, W, C = img.shape
        assert C == 3

        dmin = self.domain_min
        dmax = self.domain_max
        size = self.size
        scale = (size - 1.0) / xp.maximum(dmax - dmin, 1e-12)
        table = self.table

        out = xp.empty_like(img, dtype=xp.float32)

        # process by rows to control temp memory; 512 is a decent default
        step = 512
        for y0 in range(0, H, step):
            y1 = min(H, y0 + step)
            block = img[y0:y1].reshape(-1, 3)
            uvw = (xp.clip(block, 0.0, 1.0) - dmin) * scale  # [0, size-1]
            i0 = xp.floor(uvw).astype(xp.int32)
            f  = uvw - i0
            i1 = xp.minimum(i0 + 1, size - 1)

            iu0, iv0, iw0 = i0[:,0], i0[:,1], i0[:,2]
            iu1, iv1, iw1 = i1[:,0], i1[:,1], i1[:,2]
            fu,  fv,  fw  = f[:,0:1], f[:,1:2], f[:,2:3]

            c000 = table[iu0,iv0,iw0]
            c100 = table[iu1,iv0,iw0]
            c010 = table[iu0,iv1,iw0]
            c110 = table[iu1,iv1,iw0]
            c001 = table[iu0,iv0,iw1]
            c101 = table[iu1,iv0,iw1]
            c011 = table[iu0,iv1,iw1]
            c111 = table[iu1,iv1,iw1]

            c00 = c000*(1-fu) + c100*fu
            c10 = c010*(1-fu) + c110*fu
            c01 = c001*(1-fu) + c101*fu
            c11 = c011*(1-fu) + c111*fu
            c0  = c00*(1-fv) + c10*fv
            c1  = c01*(1-fv) + c11*fv
            c   = c0*(1-fw) + c1*fw

            out[y0:y1] = c.reshape((y1 - y0), W, 3)

        return out

# -------------------------------
# I/O helpers
# -------------------------------
def read_image_as_float(path: Path):
    """Decode HIF/HEIF/HEIC/JPG/PNG as float32 [0..1] on CPU, then move to GPU."""
    if path.suffix.lower() in (".hif", ".heif", ".heic"):
        if pillow_heif is None:
            raise RuntimeError("pillow-heif is required for HEIF. pip install pillow-heif")
        pillow_heif.register_heif_opener()
    img = Image.open(path).convert("RGB")
    arr = _np.asarray(img, dtype=_np.float32) / 255.0  # stay CPU
    return to_gpu(arr), (getattr(img, "info", {}) or {})

def save_jpeg(path: Path, rgb, quality: int = 100, exif: bytes | None = None):
    """Save sRGB JPEG; ensure CPU uint8 for Pillow."""
    arr = xp.clip(rgb.astype(xp.float32), 0.0, 1.0)
    u8 = (xp.rint(arr * 255.0)).astype(xp.uint8)
    u8_cpu = to_cpu(u8)
    img = Image.fromarray(u8_cpu, mode="RGB")
    kwargs = {"quality": int(quality), "subsampling": 0, "optimize": True}
    if exif:
        try:
            if len(exif) <= 65533:
                kwargs["exif"] = exif
            else:
                try:
                    import piexif
                    d = piexif.load(exif)
                    d["Exif"].pop(piexif.ExifIFD.MakerNote, None)
                    d["thumbnail"] = None
                    exif2 = piexif.dump(d)
                    if len(exif2) <= 65533:
                        kwargs["exif"] = exif2
                except Exception:
                    pass
        except Exception:
            pass
    img.save(path, **kwargs)

def list_cube_files(luts_dir: Path) -> list[Path]:
    return sorted(luts_dir.glob("*.cube"))

# -------------------------------
# Minimal pipeline
# -------------------------------
def process_one(
    in_path: Path,
    out_dir: Path,
    lut: CubeLUT3D,
    ev: float,
    quality: int,
    keep_exif: bool,
):
    # 1) read (CPU) -> GPU
    rgb_log, info = read_image_as_float(in_path)

    # 2) apply EV in S-Log3 linear (pre-LUT) on GPU
    lin = slog3_to_linear(rgb_log)
    if abs(ev) > 1e-8:
        lin = lin * (2.0 ** ev)
    rgb_pre_lut = xp.clip(linear_to_slog3(lin), 0.0, 1.0)

    # 3) LUT (GPU), then clamp
    rgb_disp = xp.clip(lut.apply(rgb_pre_lut), 0.0, 1.0)

    # 4) save (CPU)
    today = datetime.date.today().isoformat()
    lut_tag = in_path.stem + "_" + today
    out_name = f"{lut_tag}_{Path(lut.__dict__.get('name','lut')).stem}_EV{ev:+.2f}.jpg"
    out_path = out_dir / out_name
    exif_bytes = info.get("exif") if keep_exif else None
    save_jpeg(out_path, rgb_disp, quality=quality, exif=exif_bytes)
    print(f"✓ Saved: {out_path}")

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser(description="S-Log3 HEIF → sRGB JPEG via .cube LUT (CuPy)")
    ap.add_argument("--input-dir",  type=Path, default=Path("./Input"), help="Input folder (default ./input)")
    ap.add_argument("--output-dir", type=Path, default=Path("./Output"), help="Output folder (default ./output)")
    ap.add_argument("--luts-dir",   type=Path, default=Path("./LUTs"),  help="LUTs folder with .cube (default ./LUTs)")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality (default 100)")
    ap.add_argument("--keep-exif", action="store_true", help="Try to keep EXIF (oversize will be trimmed/dropped)")
    ap.add_argument("--recursive", action="store_true", help="Recurse input subfolders for images")
    ap.add_argument("--lut-index", type=int, default=None, help="Pick LUT by index instead of interactive choice")
    ap.add_argument("--scan-order", choices=["r_g_b","b_g_r"], default="r_g_b",
                    help="How the file was scanned to fill the 3D table (default r_g_b = R fastest)")
    ap.add_argument("--ev", type=float, default=0.0, help="Default EV (will be asked interactively after LUT selection)")
    args = ap.parse_args()

    in_dir  = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    luts_dir = args.luts_dir.resolve()

    if not in_dir.exists():
        print(f"[ERR] Input folder not found: {in_dir}"); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    lut_files = list_cube_files(luts_dir)
    if not lut_files:
        print(f"[ERR] No .cube found in: {luts_dir}"); sys.exit(2)

    # pick LUT
    if args.lut_index is None:
        print("\n== Available LUTs ==")
        for i, p in enumerate(lut_files):
            print(f"[{i}] {p.name}")
        while True:
            s = input("Enter LUT index: ").strip()
            try:
                idx = int(s)
                if 0 <= idx < len(lut_files):
                    lut_idx = idx
                    break
            except Exception:
                pass
            print("Invalid index. Try again.")
    else:
        lut_idx = args.lut_index
        if lut_idx < 0 or lut_idx >= len(lut_files):
            print(f"[ERR] LUT index out of range: {lut_idx}"); sys.exit(2)

    lut_path = lut_files[lut_idx]
    print(f"\nUsing LUT: {lut_path}")
    try:
        lut3d = CubeLUT3D.from_cube_file(lut_path)
        lut3d.name = lut_path.name
        lut3d.to_axes_rgb_inplace(args.scan_order)
    except Exception as e:
        print(f"[ERR] Failed to load LUT: {e}"); sys.exit(3)

    # Ask EV AFTER lut selection (override CLI default)
    ev = args.ev
    sv = input(f"\nExposure compensation EV (default {args.ev:+.2f}). Enter value or press Enter: ").strip()
    if sv:
        try:
            ev = float(sv)
        except Exception:
            print("Invalid EV, keep default.")

    # collect files
    exts = ("*.HIF","*.HEIF","*.HEIC","*.JPG","*.JPEG","*.PNG")
    if args.recursive:
        files = []
        for ext in exts:
            files.extend(in_dir.rglob(ext))
        files = sorted(files)
    else:
        files = []
        for ext in exts:
            files.extend(in_dir.glob(ext))
        files = sorted(files)

    if not files:
        print(f"[WARN] No images found under {in_dir}"); sys.exit(0)

    # optional: enable CuPy memory pool to reduce alloc/free overhead
    if _GPU:
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

    for f in files:
        try:
            process_one(
                in_path=f,
                out_dir=out_dir,
                lut=lut3d,
                ev=ev,
                quality=args.quality,
                keep_exif=args.keep_exif,
            )
        except Exception as e:
            print(f"[ERR] {f}: {e}")

    print(f"\nDone. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()
