# spatial-filter-tool
spatial_bandpass_filter with example(s)

## Spatial Bandpass Filter for Atmospheric Variables
### 概要 (Overview)
此工具使用 快速傅立葉轉換 (Fast Fourier Transform, FFT) 對大氣變數進行空間濾波。支援 低通 (Low-pass)與 高通 (High-pass)。
物理原理 (Physical Principles)濾波器基於響應函數 (Response function)：$$f(K) = \frac{1}{1 + (K/scale\_k)ⁿ}$$
其中 $K$ 為總波數，$scale\_k$ 為指定尺度的波數。當 $n$ 越高時，波段切割越陡峭。

### 使用方式 (Usage)
`python3 scripts/spatial_bandpass_filter.py -i input.nc -o output.bin -d 1000 -ft lowpass`
