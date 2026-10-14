# W01 / ADR-001 — MOMENT uygunluğu ve devam kararı

Tarih: 14 Ekim 2026. Durum: Koşullu devam kabul edildi.

## Bağlam

PTB-XL 1.0.3 için NORM/MI/STTC/CD/HYP çok etiketli sınıflandırma hedefleniyor.
12 kanallı, 100 Hz, 1000 örnekli giriş sözleşmesi hazır. Tüm metadata mevcut;
yalnız 1, 17 ve 39 numaralı kayıtların waveform'ları yerelde indirildi/doğrulandı.
Bu karar ilk haftanın model uygunluk deneyi içindir; model kalitesi kararı değildir.

## Kanıt

- `reports/W01-D05/model_probe.json`: MOMENT-1-small üzerinde gerçek CUDA forward ve tek optimizer adımı.
- Checkpoint revision: `411e288267f82cce86296dbe4d6c8bc533cc162f`; kaynak kod revision'ı probe raporunda.
- Girdi `[1,12,512]`, logits `[1,5]`; 512 context, 8 örneklik patch, 64 patch/kanal.
- Q/V hedefleri `named_modules()` içinden seçilen 16 Linear modülüdür.
- 57.344 LoRA ve 30.725 head parametresi eğitilebilir; 16 LoRA tensörü ve 2 head tensörü değişti.
- 35.368.133 dondurulmuş parametrenin grup hash'i aynı kaldı; base gradient almadı.
- FP32, batch 1: tepe CUDA allocated 419,89 MiB, reserved 448 MiB; bu tam eğitim bütçesi değildir.
- Yanlış target adı optimizer başlamadan reddedildi; checkpoint indirildikten sonra probe çevrimdışı da geçti.
- `reports/W01/demo.json`: aynı kayıt için PNG/metadata hash eşitliği ve yanlış zaman ölçeğinin reddi.

## Karar

MOMENT-1-small + encoder Q/V LoRA + eğitilebilir classification head ile sonraki aşamaya devam.
Önce CPU baseline kurulur; LoRA ile aynı split ve kayıt düzeyinde karşılaştırılır.
Native PyTorch yolu ilk çalışan model yolu olarak korunur. Daha büyük model/QLoRA
seçimi ancak kalite ve kaynak ölçümleriyle yapılır.

Dış pencere ve kayıt skoru birleştirme politikası henüz kesinleşmedi. Probe'un ilk
512 örneği seçmesi tam eğitim için karar sayılmaz. Eğitim ve serving aynı sürümlü
ön işlemeyi kullanacak; test fold 10 model/threshold seçimine katılmayacak.

## Bedel ve açık riskler

MOMENT PyPI 0.1.4 NumPy 1.25.2 kısıtı nedeniyle mevcut ortama uymadı; resmî kaynak
kodun sabit revision'ı kullanıldı. MOMENT Namespace config'i ile PEFT dict config'i
arasına küçük, testli bir wrapper eklendi. CUDA wheel ve checkpoint yerel disk/cache maliyeti getirir.

En riskli performans varsayımı: foundation encoder ve LoRA'nın bu ECG görevinde
baseline karşısında kalite/maliyet avantajı sağlayacağı henüz kanıtlanmadı.
En önemli değerlendirme belirsizliği: Time Series PILE ön eğitimi ile PTB-XL
örtüşmesinin sıfır olduğu doğrulanmadı. MIT model kartı kaydı, kaynak veri lisanslarının
tümünü veya klinik kullanılabilirliği tek başına doğrulamaz.
Pencereye kayıt etiketi taşımak, pencerenin kendi klinik anotasyonu olduğu anlamına gelmez.

## Süre sınırlı alternatif

Yeni bir bağımlılık/girdi/adaptör engelinde uyumluluk spike'ı toplam **3 saat** ile sınırlıdır.
Önce sabitlenmiş çalışan sürümler, batch 1, 512 context ve native PyTorch yoluna dönülür;
gerekirse açık CPU probe yapılır. CUDA geçmemişse GPU başarısı raporlanmaz.

3 saat içinde çözülemezse teknik engel ve ölçümler kaydedilir; veri/servis hattı
CPU baseline ile ilerletilir. LoRA teslimi açık eksik olarak kalır. Başka bir açık
lisanslı encoder üzerinde LoRA için ayrı, süre sınırlı spike ve yeni karar gerekir.

ONNX export deneyi ikinci haftada ayrıca **3 saat** ile sınırlıdır. Geçmezse
LoRA native PyTorch ile servis edilir, baseline için küçük ONNX yolu denenir.
Export başarısı veya LoRA-baseline üstünlüğü gerçekleşmeden iddia edilmez.

## Kararı yeniden değerlendirme koşulları

Hasta ayrımı veya sinyal sözleşmesi bozulursa; save/load parity başarısızsa;
validation'da sınıf bazlı kalite/latency/bellek hedefleri karşılanmazsa; checkpoint
lisansı veya ön eğitim örtüşmesi hakkındaki yeni kanıt kullanım varsayımını değiştirirse.
