# =============================================================================================
# ==== INFOMATION ========
# ========================
# 檔名: spatial_bandpass_filter.py
# 功能: 對大氣變數進行空間濾波 (高通/低通/帶通)
# 作者: YakultSmoothie and Claude(CA)
# 建立日期: 2024-12-06
# 更新日期: 2024-12-09 - YakultSmoothie and Claude(CA)
#           增加 -np 參數，可選擇不輸出比較圖檔
#
# Description:
#   此程式讀取netCDF檔案，對指定變數進行空間濾波，
#   支援高通、低通和帶通三種濾波方式。
#   響應函數(在create_filter中)決定濾波的波段與強度
#   並將結果輸出為二進制檔案。濾波使用FFT方法。
# ============================================================================================

import numpy as np
import xarray as xr
import argparse
from datetime import datetime
import sys
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='空間濾波程序 - 對大氣變數進行空間濾波',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 1. 使用預設參數進行低通濾波
     python3 spatial_bandpass_filter.py

  # 2. 對850hPa的溫度場進行高通濾波，保留小於1000km的訊號
     python3 spatial_bandpass_filter.py -V t -Z 850 -d 1000 -ft highpass

  # 3. 處理特定時間的數據，並指定輸出檔案
     python3 spatial_bandpass_filter.py -T 2024111300 -o output.bin

  # 4. 使用帶通濾波，保留接近2000km的尺度
     python3 spatial_bandpass_filter.py -d 2000 -ft bandpass

  # 5. 不輸出比較圖檔案
     python3 spatial_bandpass_filter.py -np

註記:
  - 輸入檔案必須是netCDF格式
  - 輸出為二進制檔案，資料型態為32位元浮點數
  - 預設會產生濾波前後的比較圖，可用-np關閉

作者: YakultSmoothie and Claude(CA)
建立日期: 2024-12-09 [v1.0]

""")

    # Input/Output arguments
    io_group = parser.add_argument_group('輸入輸出選項')
    io_group.add_argument('-i', '--input',
                         default='/jet/ox/DATA/ERA5/M09-11/2024.pre.nc',
                         help='輸入nc檔案路徑')
    io_group.add_argument('-o', '--output',
                         default='./out-sbf.bin',
                         help='輸出bin檔案路徑')

    # Data selection arguments
    data_group = parser.add_argument_group('資料選擇選項')
    data_group.add_argument('-V', '--variable',
                           default='u',
                           help='要濾波的變數 (例如: u, v, t)')
    data_group.add_argument('-T', '--time',
                           default='2024111300',
                           help='時間 (格式: YYYYMMDDHH)')
    data_group.add_argument('-L', '--level',
                           type=float,
                           default=850,
                           help='氣壓層 (單位: hPa)')

    # Filter arguments
    filter_group = parser.add_argument_group('濾波選項')
    filter_group.add_argument('-d', '--scale',
                            type=float,
                            default=600,
                            help='濾波尺度 (單位: km)')
    filter_group.add_argument('-ft', '--filter_type',
                            choices=['lowpass', 'highpass', 'bandpass'],
                            default='lowpass',
                            help='''濾波器類型:
    lowpass  - 保留大於指定尺度的訊號,
    highpass - 保留小於指定尺度的訊號,
    bandpass - 保留接近指定尺度的訊號''')

    # Output control arguments
    output_group = parser.add_argument_group('輸出控制選項')
    output_group.add_argument('-np', '--no-plot',
                            action='store_true',
                            help='不輸出比較圖檔案')

    return parser.parse_args()

def read_netcdf_data(filename, variable, time_str, level):
    """讀取netCDF數據"""
    try:
        time = datetime.strptime(time_str, '%Y%m%d%H')
        ds = xr.open_dataset(filename)
        
        data = ds[variable].sel(
            valid_time=time,
            pressure_level=level,
            method='nearest'    # 最近鄰插值法匹配數據
        ).values
        
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        print(f"INFO [read_netcdf_data]:")
        print(f"[read_netcdf_data] Domain range: lon [{lons.min():.2f}°E - {lons.max():.2f}°E], lat [{lats.min():.2f}°N - {lats.max():.2f}°N]")
        print(f"[read_netcdf_data] Grid spacing: dlon = {np.mean(np.diff(lons)):.3f}°, dlat = {np.mean(np.diff(lats)):.3f}°")
        print(f"[read_netcdf_data] Variable: {variable} at {level}hPa")
        print(f"[read_netcdf_data] Time: {time}")
        print(f"[read_netcdf_data] Data range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}] {ds[variable].units}")
        
        ds.close()
        return data, lats, lons
        
    except Exception as e:
        print(f"錯誤: 讀取netCDF檔案失敗 - {str(e)}")
        sys.exit(1)

def create_filter(K, scale_km, filter_type):
    """
    創建空間濾波器
    K: 空間頻率矩陣，代表在頻域中每個點對應的總空間頻率
    scale_km: 目標空間尺度，單位是公里。
    filter_type: 用K與scale_k決定濾波權重。決定訊號的保留方式。
    """
    
    scale_k = 1.0 / scale_km
    
    # 增加選項，可更接近的訊號在把能量留下來
    if filter_type == 'lowpass':
        return 1.0 / (1.0 + (K/scale_k)**8) 
        # 低通濾波器: 保留大於scale_km的尺度。
        # K << scale_k 時接近 1（保留訊號），在 K >> scale_k 時接近 0（濾除訊號）。
        # 次方數(n)越高代表著更明確的頻率邊界。
    elif filter_type == 'highpass':
        return (K/scale_k)**2 / (1.0 + (K/scale_k)**8)  
        # 高通濾波器: 保留小於scale_km的尺度
        # K >> scale_k 時接近 1（保留訊號），在 K << scale_k 時接近 0（濾除訊號）。
    elif filter_type == 'bandpass':
        return np.exp(-((K - scale_k) / (0.1 * scale_k))**2) 
        # 帶通濾波器: 保留接近scale_km的尺度
        # 濾波器的響應呈高斯分佈，中心頻率是 scale_k，標準差是 0.1 * scale_k
    else:
        raise ValueError(f"未知的濾波器類型: {filter_type}")

def apply_spatial_filter(data, lats, lons, scale_km, filter_type, no_plot=False):
    """應用空間濾波"""
    print(f"INFO [apply_spatial_filter]:")
    try:
        # 計算網格間距，考慮球面幾何效應
        dy = np.abs(np.mean(np.diff(lats))) * 111.19  # 緯度間距(km)

        # 計算經度間距，隨緯度變化 (dx = R * cos(lat) * dlon)
        dx_base = np.abs(np.mean(np.diff(lons))) * 111.19  # 赤道上的經度間距(km)
        cos_lats = np.cos(np.deg2rad(lats))  # 各緯度的餘弦值

        # 輸出網格資訊
        print(f"[apply_spatial_filter] Grid spacing:")
        print(f"  dy = {dy:.2f} km (constant)")
        print(f"  dx = {dx_base:.2f} km at equator")
        print(f"  dx = {dx_base * np.cos(np.deg2rad(np.min(lats))):.2f} km at lat {np.min(lats):.1f}°")
        print(f"  dx = {dx_base * np.cos(np.deg2rad(np.max(lats))):.2f} km at lat {np.max(lats):.1f}°")

        # 計算FFT
        fft_data = fft2(data)  # 2D 快速傅立葉轉換，將空間域的資料轉換到頻域，揭示資料在不同空間頻率上的貢獻。
                               # 會輸出一個複數（實數+虛數）矩陣
                               # 取振幅（magnitude）：np.abs(fft_data)
                               # 取相位（phase）：np.angle(fft_data)
                               # 取功率譜（power spectrum）：np.abs(fft_data)**2
                               # 移動頻率零點到中心：np.fftshift(fft_data)

        # 生成頻率網格，考慮緯向變化的dx
        ky = fftfreq(data.shape[0], dy)       # 計算空間頻率（spatial frequency）= y方向的波數 
        kx = fftfreq(data.shape[1], dx_base)  # x方向的波數
        KX, KY = np.meshgrid(kx, ky)          # 創建2D波數網格
                                              # KX矩陣的每一列都是kx的複製
                                              # KY矩陣的每一行都是ky的複製
                                              # 每個網格點(i,j)都有對應的(KX[i,j], KY[i,j])波數值
        KX = KX / cos_lats[:, np.newaxis]     # 隨緯度調整KX

        # 計算總波數
        K = np.sqrt(KX**2 + KY**2)

        # 應用濾波
        filter_response = create_filter(K, scale_km, filter_type) # 建立濾波器(filter_response)
                                                                  # 值介於0和1之間，決定每個波數要保留多少能量 
        filtered_fft = fft_data * filter_response     # 用濾波器乘以傅立葉係數，實現頻率選擇
        filtered_data = np.real(ifft2(filtered_fft))  # 濾波結果，取filtered_fft的實數部位
                                                      # 執行2D反傅立葉變換

        # 增加濾波資訊輸出
        filter_desc = {
            'lowpass': f'>{scale_km}km',
            'highpass': f'<{scale_km}km',
            'bandpass': f'~{scale_km}km'
        }

        print(f"[apply_spatial_filter] Filter type: {filter_type} ({filter_desc[filter_type]})")
        print(f"[apply_spatial_filter] Original data range: [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
        print(f"[apply_spatial_filter] Filtered data range: [{np.nanmin(filtered_data):.2f}, {np.nanmax(filtered_data):.2f}]")

        # 創建比較圖 (只在no_plot=False時執行)
        if not no_plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            im1 = ax1.contourf(lons, lats, data, levels=20)
            ax1.set_title('Original Field')
            plt.colorbar(im1, ax=ax1)
            ax1.set_xlabel('Longitude (°E)')
            ax1.set_ylabel('Latitude (°N)')
            im2 = ax2.contourf(lons, lats, filtered_data, levels=20)
            ax2.set_title(f'Filtered Field ({filter_desc[filter_type]})')
            plt.colorbar(im2, ax=ax2)
            ax2.set_xlabel('Longitude (°E)')

            plt.suptitle(f'Spatial {filter_type.capitalize()} Filter Comparison\n' +
                        f'Scale: {scale_km}km, Grid: dx(at e.q.)={dx_base:.1f}km, dy={dy:.1f}km\n' +
                        f'Domain: {lons.min():.1f}°E-{lons.max():.1f}°E, ' +
                        f'{lats.min():.1f}°N-{lats.max():.1f}°N',
                        y=1.08)

            plt.tight_layout()
            plt.savefig(f'log.sbf-{filter_type}_comparison.png', bbox_inches='tight', dpi=150)
            print(f"  Comparison plot saved as: log.sbf-{filter_type}_comparison.png")
            plt.close()

        return filtered_data

    except Exception as e:
        print(f"錯誤: 濾波過程失敗 - {str(e)}")
        sys.exit(1)


def save_binary(data, output_file):
    """保存為二進制檔案"""
    try:
        data = data.astype(np.float32)
        with open(output_file, 'wb') as f:
            data.tofile(f)
            
        print(f"INFO (save_binary):")
        print(f"  成功寫入檔案: {output_file}")
        print(f"  檔案大小: {data.nbytes} bytes")
        print(f"  數據維度: {data.shape}")
        
    except Exception as e:
        print(f"錯誤: 寫入二進制檔案失敗 - {str(e)}")
        sys.exit(1)

def main():
    """主程序"""
    args = parse_arguments()

    # 讀取數據
    print(f"\n====== START of spatial_bandpass_filter.py ======")
    print(f"\n( I ) 讀取檔案: {args.input}")
    data, lats, lons = read_netcdf_data(
        args.input, args.variable, args.time, args.level
    )
    print(f"( I.end ) 數據維度: {data.shape}")

    # 應用濾波
    print(f"\n( II ) 進行 {args.scale}km 尺度的{args.filter_type}濾波...")
    filtered_data = apply_spatial_filter(data, lats, lons, args.scale, 
                                       args.filter_type, no_plot=args.no_plot)

    # 保存結果
    print(f"\n( III ) 寫入結果到: {args.output}")
    save_binary(filtered_data, args.output)
    print(f"\n====== END of spatial_bandpass_filter.py ======\n")

if __name__ == "__main__":
    main()

# ============================================================================================
