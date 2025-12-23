✏️ Karakalem Efekti – Görüntü İşleme Projesi

Bu proje, Python ve Streamlit kullanılarak geliştirilmiş basit ve etkili bir görüntü işleme uygulamasıdır. Kullanıcı tarafından yüklenen bir fotoğraf, OpenCV teknikleri kullanılarak karakalem (sketch) efektine dönüştürülür.

🚀 Özellikler

JPG, PNG ve WEBP formatlarında fotoğraf yükleme

Gri tonlama (Grayscale)

CLAHE ile kontrast artırma

Gaussian Blur uygulama

Karakalem (Sketch) efekti oluşturma

Orijinal ve işlenmiş görüntüyü yan yana gösterme

Kullanıcı dostu Streamlit arayüzü

🛠️ Kullanılan Teknolojiler

Python

Streamlit

OpenCV (cv2)

NumPy

Pillow (PIL)

📂 Proje Yapısı

karakalem-efekti
├── app.py
└── README.md

⚙️ Kurulum ve Çalıştırma

Gerekli kütüphaneleri yüklemek için:

pip install streamlit opencv-python numpy pillow

Uygulamayı çalıştırmak için:

streamlit run app.py

🖼️ Uygulama Nasıl Çalışır?

Kullanıcı bir fotoğraf yükler

Fotoğraf gri tonlamaya çevrilir

CLAHE yöntemi ile kontrast artırılır

Görüntü terslenir ve bulanıklaştırılır

cv2.divide yöntemi ile karakalem efekti elde edilir

Keskinleştirme filtresi uygulanır

Sonuç ekranda gösterilir

🎯 Amaç

Bu proje, görüntü işleme mantığını öğrenmek, OpenCV kütüphanesini pratikte kullanmak ve Streamlit ile basit bir arayüz geliştirmek amacıyla hazırlanmıştır.

👤 Geliştirici

Hasan Şimşek
E-posta: simsekhasan2112@gmail.com
