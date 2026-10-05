# Video qualification fixture

`red-blue.mp4` is a generated two-second, 64 x 64 H.264 video: one second of
solid red, then one second of solid blue, at four frames per second. No audio,
external source material, or model output is included. It tests native temporal
video input separately from image input.

Reproduce with FFmpeg:

```bash
ffmpeg -f lavfi -i color=c=red:s=64x64:r=4:d=1 \
  -f lavfi -i color=c=blue:s=64x64:r=4:d=1 \
  -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0[v]' -map '[v]' \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart red-blue.mp4
```
