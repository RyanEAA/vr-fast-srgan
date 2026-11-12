# Results

Machine: macOS 15.3.1 (Darwin 24.3.0) · Apple M4 Mac mini, 16 GB · Python 3.14.8 · torch 2.15.0  
Test clip: `clips/pano.mp4` (three.js sample, 1920×1080, 7.5 s, 25 fps). Low-res stream = ¼ width and height. All streams H.264 CRF 23 unless noted.

## Bandwidth, speed and quality

Quality is measured on the **visible 120°×120° region** against the original video (higher PSNR/SSIM is better, lower flicker is better). Time per frame is the upscaling step only (decode time for the two full-res streams). Real time = under 40 ms (25 fps).

| Technique | Stream kbps | GB/hour | ms/frame | Real time | PSNR dB | SSIM | Flicker |
|---|---:|---:|---:|:---:|---:|---:|---:|
| Full-res stream, whole 360° (baseline) | 3069 | 1.38 | 1.0 | ✓ | 42.3 | 0.983 | 1.43 |
| Full-res stream, visible region only | 1046 | 0.47 | 0.3 | ✓ | 40.6 | 0.980 | 1.73 |
| Low-res stream + no upscaling | 531 | 0.24 | 0.5 | ✓ | 28.9 | 0.866 | 3.97 |
| Low-res stream + bicubic | 531 | 0.24 | 1.0 | ✓ | 31.0 | 0.921 | 3.68 |
| Low-res stream + Lanczos | 531 | 0.24 | 2.2 | ✓ | 31.1 | 0.922 | 3.69 |
| Low-res stream + Fast-SRGAN (fp32) | 531 | 0.24 | 168.8 | ✗ | 27.5 | 0.809 | 6.24 |
| Low-res stream + Fast-SRGAN (fp16) | 531 | 0.24 | 132.0 | ✗ | 27.5 | 0.809 | 6.24 |
| Low-res stream + Fast-SRGAN on visible region (fp16) | 531 | 0.24 | 31.6 | ✓ | 28.1 | 0.830 | 5.76 |

Full-res stream squeezed to the low-res bitrate (553 kbps): whole-frame PSNR 30.0 dB, vs 31.8 dB for low-res + bicubic at the same bandwidth.

## Resources while playing at 25 fps

One fresh process per technique: decode the stream → upscale → full-res frame. GPU numbers are above the idle level (Apple GPU via `ioreg`; not available on other platforms).

| Technique | CPU (% of 1 core) | RAM peak MB | GPU memory added MB | GPU busy % |
|---|---:|---:|---:|---:|
| Full-res stream, whole 360° (baseline) | 45 | 262 | 0 | 12 |
| Full-res stream, visible region only | 30 | 251 | 1 | 11 |
| Low-res stream + no upscaling | 37 | 205 | 4 | 10 |
| Low-res stream + bicubic | 53 | 193 | 1 | 11 |
| Low-res stream + Lanczos | 88 | 195 | 29 | 12 |
| Low-res stream + Fast-SRGAN (fp16) | 26 | 633 | 1051 | 92 |
| Low-res stream + Fast-SRGAN on visible region (fp16) | 57 | 613 | 1063 | 80 |

## Fast-SRGAN time vs. how much of the view is upscaled

Large crops run as 160×160 low-res tiles to keep GPU memory around 1 GB (whole 4K/8K frames in one pass exhausted 16 GB).

| Source | Region | Input → output | fp32 ms | fp16 ms | fp16 fps |
|---|---|---|---:|---:|---:|
| 1080p clip (this test) | full frame | 480x240 → 1920x960 | 141.9 | 107.5 | 9.3 |
| 1080p clip (this test) | viewport 120x120 deg (90 deg FOV + head-motion margin) | 160x160 → 640x640 | 30.6 | 23.4 | 42.8 |
| 1080p clip (this test) | viewport 90x90 deg | 120x120 → 480x480 | 17.2 | 13.2 | 75.6 |
| 1080p clip (this test) | fovea 20x20 deg (eye tracking) | 27x27 → 108x108 | 1.3 | 1.3 | 776.4 |
| 4K 360 video | viewport 120x120 deg (90 deg FOV + head-motion margin) | 320x320 → 1280x1280 | 126.5 | 94.9 | 10.5 |
| 4K 360 video | viewport 90x90 deg | 240x240 → 960x960 | 72.1 | 53.8 | 18.6 |
| 4K 360 video | fovea 20x20 deg (eye tracking) | 53x53 → 212x212 | 3.5 | 3.0 | 333.0 |
| 8K 360 video | viewport 120x120 deg (90 deg FOV + head-motion margin) | 640x640 → 2560x2560 | 490.0 | 373.9 | 2.7 |
| 8K 360 video | viewport 90x90 deg | 480x480 → 1920x1920 | 275.7 | 210.2 | 4.8 |
| 8K 360 video | fovea 20x20 deg (eye tracking) | 107x107 → 428x428 | 14.2 | 11.1 | 90.3 |
