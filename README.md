# Introduction Detector dengan Kamera

Aplikasi deteksi perkenalan diri menggunakan kamera dan gesture tangan dengan output suara menggunakan Google Text-to-Speech.

## Fitur

- Deteksi gesture tangan menggunakan MediaPipe
- Text-to-Speech dalam Bahasa Indonesia menggunakan gTTS
- Real-time detection menggunakan webcam
- Interface visual yang user-friendly

## Kata-kata yang Didukung

1. **1 jari** → "perkenalkan"
2. **2 jari** → "nama saya"
3. **3 jari** → "rendy"
4. **4 jari** → "nayogi"
5. **5 jari** → "pramudya"
6. **6 jari** (kelima jari + thumb) → "terima kasih"

## Instalasi

1. Install Python 3.8 atau lebih baru

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Cara Menggunakan

1. Jalankan aplikasi:

```bash
python main.py
```

2. Tunjukkan jumlah jari ke kamera sesuai dengan kata yang ingin diucapkan

3. Kontrol keyboard:
   - **Q** → Keluar dari aplikasi
   - **R** → Reset
   - **S** → Ucapkan kata sesuai gesture saat ini

## Cara Kerja

1. Aplikasi menggunakan MediaPipe untuk mendeteksi tangan dan menghitung jumlah jari yang terangkat
2. Setiap jumlah jari dipetakan ke kata tertentu
3. Ketika gesture terdeteksi, aplikasi akan memutar audio yang sudah di-generate menggunakan gTTS
4. Audio disimpan di folder `audio/` untuk performa yang lebih baik

## Troubleshooting

- Pastikan webcam Anda terhubung dan berfungsi
- Gunakan pencahayaan yang baik untuk deteksi yang lebih akurat
- Posisikan tangan dengan jelas di depan kamera
- Jika audio tidak terdengar, periksa volume sistem Anda

## Requirements

- Python 3.8+
- Webcam
- Koneksi internet (untuk pertama kali generate audio)
