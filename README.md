# Intel-Image-Classification---Deep-Learning-Projes

**Akbank Derin Öğrenme Bootcamp | Eylül 2025**

Bu proje, Intel Image Classification veri seti kullanılarak CNN (Convolutional Neural Network) mimarisi ile 6 farklı doğal ve yapay ortamın sınıflandırılması üzerine çalışılmıştır. Proje kapsamında modern derin öğrenme teknikleri, overfitting kontrolü, model yorumlanabilirliği ve hiperparametre optimizasyonu uygulanmıştır.

##  Proje Özeti

### Hedef
Görüntüleri 6 farklı kategoriye ayırmak:
- **Buildings** (Binalar)
- **Forest** (Orman)
- **Glacier** (Buzul)
- **Mountain** (Dağ)
- **Sea** (Deniz) 
- **Street** (Sokak)

### Ana Sonuçlar
- **Test Accuracy**: %82.03 
- **F1-Score**: 0.8203
- **Model Boyutu**: 3.2 MB (850,598 parametre)
- **Overfitting**: Başarıyla kontrol edildi

##  Veri Seti Bilgileri

**Intel Image Classification Dataset** - Kaggle'dan alınan popüler bir görüntü sınıflandırma veri seti.

### Veri Dağılımı
| Sınıf | Eğitim | Test |
|-------|---------|------|
| Mountain | 2,512 | 525 |
| Street | 2,382 | 501 |
| Buildings | 2,191 | 437 |
| Sea | 2,274 | 510 |
| Forest | 2,271 | 474 |
| Glacier | 2,404 | 553 |
| **Toplam** | **14,034** | **3,000** |

- Görüntü boyutu: 224x224 piksel
- Batch size: 32
- Validation split: %20

##  Kullanılan Yöntemler

### 1. Veri Ön İşleme & Artırma
Overfitting'i önlemek ve model genelleme yeteneğini artırmak için güçlü data augmentation teknikleri uygulandı:

```python
# Uygulanan Dönüşümler
- Rotation: ±25°
- Width/Height Shift: ±20%
- Horizontal Flip: Evet
- Zoom: ±20%
- Brightness: %70-130
- Shear Transformation: ±15°
```

### 2. CNN Model Mimarisi
Overfitting'e karşı gelişmiş regularization teknikleriyle tasarlanmış özel CNN mimarisi:

**Temel Bileşenler:**
- 4 adet Convolutional blok (32, 64, 128, 256 filtre)
- Batch Normalization (her conv bloktan sonra)
- Dropout katmanları (0.3 - 0.6 arası)
- L2 Regularization (0.002)
- Global Average Pooling
- Dense katmanlar (512, 256 nöron)

**Aktivasyon Fonksiyonları:**
- Hidden layers: ReLU
- Output layer: Softmax (6 sınıf için)

### 3. Eğitim Stratejisi
Model eğitimi için akıllı callback sistemi kullanıldı:

- **Learning Rate**: 0.0005 (kontrollü öğrenme)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Early Stopping**: Patience=12 (val_loss takibi)
- **Learning Rate Reduction**: Factor=0.2, Patience=5
- **Model Checkpoint**: En iyi F1-score'u kaydetme

### 4. Regularization Teknikleri
Overfitting'i önlemek için çoklu strateji:
- Dropout oranları: %30-60 arası
- L2 regularization: 0.002
- Güçlü data augmentation
- Early stopping
- Global average pooling

##  Model Performansı

### Sınıf Bazında Sonuçlar
| Sınıf | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Buildings | 0.905 | 0.700 | 0.790 | 437 |
| Forest | 0.885 | 0.971 | 0.926 | 474 |
| Glacier | 0.806 | 0.772 | 0.789 | 553 |
| Mountain | 0.762 | 0.787 | 0.774 | 525 |
| Sea | 0.767 | 0.882 | 0.820 | 510 |
| Street | 0.839 | 0.808 | 0.823 | 501 |

### Genel Metrikler
- **Accuracy**: 82.03%
- **Macro Avg F1**: 0.820
- **Weighted Avg F1**: 0.819
- **Overfitting Durumu**: Kontrol altında 

**En İyi Sınıf**: Forest (%92.6 F1-score)  
**En Zorlu Sınıf**: Mountain (%77.4 F1-score)

##  Model Yorumlanabilirliği

### Grad-CAM Analizi
Modelin hangi görüntü bölgelerine odaklandığını anlamak için Grad-CAM (Gradient-weighted Class Activation Mapping) tekniği uygulandı. Bu sayede:

- Model kararlarının görselleştirilmesi
- Yanlış sınıflandırmaların nedenlerinin anlaşılması
- Model güvenilirliğinin değerlendirilmesi sağlandı

### Confusion Matrix İçgörüleri
- Forest sınıfı en yüksek recall'a sahip (0.971)
- Buildings sınıfının precision'ı en yüksek (0.905)
- Mountain ve Glacier arasında karışıklık gözlemlendi

##  Hiperparametre Optimizasyonu

Sistemli hiperparametre testi gerçekleştirildi:

**Test Edilen Parametreler:**
- Dropout oranları: 0.3, 0.5
- Learning rate: 0.001, 0.0005
- Optimizer: Adam, RMSprop
- Batch size: 32, 64
- L2 regularization: 0.001, 0.002

**Sonuç**: Mevcut konfigürasyon optimal performans sağladı.

##  Transfer Learning Karşılaştırması

Custom CNN yanında transfer learning de test edildi:
- Base model alternatifi oluşturuldu
- 5 epoch ile hızlı eğitim yapıldı
- Custom CNN ile performans karşılaştırıldı

##  Teknik Detaylar ve Öğrendiklerim

### Overfitting Kontrolü
Bu projede en büyük meydan okuma overfitting'ti. Çözüm için:
- Dropout oranlarını %30'dan %60'a çıkardım
- L2 regularization katsayısını 0.001'den 0.002'ye artırdım
- Data augmentation'ı güçlendirdim (shear transformation ekledim)
- Learning rate'i 0.001'den 0.0005'e düşürdüm

### Model Mimarisi Kararları
- **Global Average Pooling**: Flatten yerine GAP kullanarak parametre sayısını azalttım
- **Batch Normalization**: Her conv bloktan sonra ekleyerek gradient flow'u iyileştirdim
- **Progressive Dropout**: Ağın derinleştikçe dropout oranını artırdım

### F1-Score Custom Metric
Accuracy yanında F1-score da izledim çünkü:
- Çok sınıflı probleme daha uygun
- Precision ve recall dengesini gösteriyor
- Model selection için daha güvenilir

##  Kullanılan Teknolojiler

- **Framework**: TensorFlow/Keras 2.18.0
- **GPU**: Tesla T4 (Kaggle ortamı)
- **Veri İşleme**: ImageDataGenerator, OpenCV
- **Görselleştirme**: Matplotlib, Seaborn
- **Metrikler**: Scikit-learn
- **Model Yorumlama**: Grad-CAM (custom implementation)

##  Proje Yapısı

```
📂 Notebook İçeriği
├── 1. Veri Keşfi ve İstatistikler
├── 2. Veri Görselleştirme  
├── 3. Data Augmentation
├── 4. CNN Model Mimarisi
├── 5. Model Eğitimi
├── 6. Performans Değerlendirmesi
├── 7. Confusion Matrix Analizi
├── 8. Grad-CAM Görselleştirmesi
├── 9. Hiperparametre Optimizasyonu
├── 10. Transfer Learning
├── 11. Sonuçlar ve İyileştirmeler
└── 12. Model Kaydetme
```
**Projenin tamamı projenin çok büyük olmasından ötürü 4 parçaya ayrılarak githuba yüklenmiştir.**

##  Gelecek Çalışmalar

### Kısa Vadeli İyileştirmeler
- **EfficientNet** veya **ResNet** ile transfer learning
- **Test Time Augmentation** (TTA) uygulama
- **Ensemble methods** ile model kombinasyonu
- **CutMix** ve **MixUp** augmentation teknikleri

### Orta Vadeli Hedefler
- **Streamlit** ile web uygulaması geliştirme
- **Docker** ile containerization
- **MLflow** ile model versioning
- **Gradio** ile interaktif demo

### Uzun Vadeli Vizyon
- **Object detection** ile sahne analizi
- **Semantic segmentation** için pixel-level sınıflandırma
- **Real-time inference** için model optimization
- **Mobile deployment** için quantization

##  Proje Linkleri

**Kaggle Notebook**: https://www.kaggle.com/code/szeynepdrk/intel-image-classificatin

**Kaggle Veriseti**: https://www.kaggle.com/datasets/puneet6060/intel-image-classification


##  Sonuç

Bu proje kapsamında Intel Image Classification veri seti üzerinde %82 doğruluk oranına sahip, dengeli ve yorumlanabilir bir CNN modeli geliştirildi. Overfitting başarıyla kontrol edildi ve model gerçek dünya uygulamaları için hazır hale getirildi.

Proje boyunca derin öğrenme pipeline'ının tüm aşamalarını deneyimleme fırsatı buldum: veri ön işlemeden model deployment'a kadar. Özellikle regularization teknikleri, callback sistemi ve model yorumlanabilirliği konularında derinlemesine öğrendim.

---

**Geliştirici**: Zeynep Sıla Durak  
**Bootcamp**: Akbank Derin Öğrenme Bootcamp  
**Tarih**: Eylül 2025  
**Framework**: TensorFlow/Keras
