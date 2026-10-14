# PTB-XL — veri ve görev sözleşmesi

Durum: Gün 2 için uygulandı ve doğrulandı. Tarih: 14 Ekim 2026.

## Kaynak ve sürüm

- Veri: PTB-XL, a large publicly available electrocardiography dataset.
- Sağlayıcı/yazarlar: Wagner, Strodthoff, Bousseljot, Samek ve Schaeffter; PhysioNet.
- Sabit sürüm: **1.0.3** (9 Kasım 2022).
- Sürüm DOI: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).
- [Resmî veri sayfası](https://physionet.org/content/ptb-xl/1.0.3/).
- [Kanonik dosya kaynağı](https://physionet.org/files/ptb-xl/1.0.3/).
- İndirme: resmî sayfada belirtilen `physionet-open` açık S3 aynası;
  her dosyanın gerçek ve kanonik URL'si `provenance.json` içinde kayıtlı.
- 1.0.3, önceki sürümlerdeki yinelenen kayıt/etiket düzeltmelerini içerir;
  başka sürümün metadata veya etiket tablosu bu sözleşmeye karıştırılmaz.

## Lisans ve atıf

Veri lisansı [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Lisans; kopyalama, paylaşma, dönüştürme ve ticari kullanıma izin verir.
Paylaşımda uygun yazar/kaynak atfı, lisans bağlantısı ve yapılan değişikliklerin
belirtilmesi gerekir; kaynak sahiplerinin projeyi onayladığı ima edilmez.
Lisansın izinlerini daraltan ek hukuki/teknik kısıtlar konulmaz.

Bu projede kaynak veri değiştirilmez; diagnostic superclass hedefleri türetilir,
boş hedefli kayıtlar eğitim etiket listesinden çıkarılır ve yalnızca üç waveform
yerel örnek olarak seçilir. Lisans metni ham verilerle birlikte saklanır.
Veri lisansı uygulama kodunun lisansını otomatik belirlemez.

## Görev

Girdi: bir kaydın 100 Hz örneklenmiş, 10 saniyelik, 12 derivasyonlu ECG sinyali;
gelecekteki model girdisinin zaman/kanal düzeni `(1000, 12)` olacaktır.
Bugün waveform dosyaları indirildi ve WFDB header bilgileri doğrulandı;
sayısal waveform yükleme, ön işleme ve model eğitimi henüz yapılmadı.

Çıktı: aynı kayda ait beş diagnostic superclass için sabit sıralı multi-hot hedef.
Hedef kaynağı `ptbxl_database.csv` içindeki `scp_codes`; eşleme kaynağı
`scp_statements.csv` içindeki `diagnostic == 1.0` ve `diagnostic_class` alanlarıdır.
Rapor metni/demografik alanlar model girdisine dahil edilmez.

| İndeks | Etiket | Kaynaktaki karşılığı | Pozitif kayıt |
| --- | --- | --- | ---: |
| 0 | NORM | Normal ECG | 9514 |
| 1 | MI | Myocardial Infarction | 5469 |
| 2 | STTC | ST/T Change | 5235 |
| 3 | CD | Conduction Disturbance | 4898 |
| 4 | HYP | Hypertrophy | 2649 |

Etiket sırası **`[NORM, MI, STTC, CD, HYP]`** değişmez. Bir kayıtta birden çok
pozitif bulunabilir; aynı superclass'a eşlenen kodlar tek pozitif oluşturur.
NORM sadece açık NORM kodundan üretilir. Başka pozitiflerle birlikte gelirse
kaynak etiket sessizce düzeltilmez. Hedefteki 0, bu politika altında etiket
atanmadığını gösterir; klinik yokluk kanıtı olarak yorumlanmaz.

## Etiket politikası

Politika sürümü: **`ptbxl-superclass-presence-v1`**.
Makine tarafından okunabilir sözleşme: `label_schema.json`.

1. `scp_codes`, güvenli literal parser ile sözlüğe çevrilir; `eval` kullanılmaz.
2. Tekrarlı anahtar, bilinmeyen kod, sayısal olmayan/boolean/nonfinite veya
   `[0,100]` dışındaki güven değeri doğrulama hatasıdır. Sessizce atılmaz.
3. Resmî tablo 71 bilinen kod içerir; 44 diagnostic kod beş superclass'a eşlenir.
   Bilinen non-diagnostic kodlar önce doğrulanır, sonra hedef üretiminde yok sayılır.
4. Kabul edilen likelihood: **0–100 aralığında sonlu sayılar**. Threshold yoktur;
   diagnostic kodun varlığı superclass pozitifliği için yeterlidir. Bu, resmî
   `example_physionet.py` içindeki kod-varlığı eşlemesini izler.
5. `0` likelihood kaynağa göre bilinmeyen güvendir; negatif etiket değildir.
   Bu politika `0` değerli diagnostic kodu da dahil eder. Low-confidence kodların
   dahil edilmesi açık bir veri politikasıdır; başka eşik yeni politika sürümü ister.
6. Diagnostic superclass bulunmazsa sonuç `null` ve dışlama nedeni
   `no_diagnostic_superclass` olur. Kayıt `[0,0,0,0,0]` veya NORM yapılmaz.
7. Kabul edilen hedefler sabit sırada 0/1 vektöre dönüştürülür.

Metadata'da görülen likelihood değerleri: 0, 10, 14, 15, 30, 34, 35, 50, 80, 100.
Değerler belli bir küçük kümeye zorlanmaz. Kod/güven doğrulaması known rhythm/form
kodlarına da uygulanır.

Politikayı değiştirmek veri kümesini değiştirir: yeni politika sürümü, yeniden
üretilmiş etiketler ve yeni provenance gerekir. İşlem mevcut farklı schema'yı
sessizce üzerine yazmaz. Sıralama metadata'nın satır/kod sırasından bağımsızdır.

## Tam metadata denetimi

Bu sayılar üç örnekten değil, sürümün **tüm metadata kayıtlarından** hesaplandı:

| Ölçüm | Sonuç |
| --- | ---: |
| İncelenen kayıt | 21799 |
| Dahil edilen | 21388 |
| Boş diagnostic hedef nedeniyle dışlanan | 411 |
| Çok etiketli dahil edilen kayıt | 5144 |
| Bilinmeyen SCP kodu | 0 |

Dışlanan kayıt sayısı, `strat_fold` sırasıyla 1–10:
`[36,43,42,44,39,44,42,44,37,40]`.
Her patient_id'nin tek fold içinde kalması kontrol edildi; geçti.
Etiket başına sayılar toplanınca kayıt sayısından fazla çıkabilir.

İleride kullanılacak kaynak önerisi: fold 1–8 eğitim, 9 doğrulama, 10 test.
Bugün eğitim yapılmadı. Metadata'nın sağladığı hasta bazlı fold'lar korunur;
aynı hastanın kayıtlarının train/test'e karışması önlenir.

## Küçük waveform örneği

Seçim deterministiktir: her rol için en küçük ecg_id. Temsil gücü olan rastgele
örnek veya performans değerlendirme seti olarak kullanılmaz.

| ecg_id | Rol | Hedef |
| ---: | --- | --- |
| 1 | Normal | `[1,0,0,0,0]` |
| 39 | Çok etiketli | `[0,1,1,0,0]` |
| 17 | Dışlanan | `null` |

39 numaralı kayıtta IMI:15 MI üretir; LNGQT/NST_/DIG STTC üretir.
17 numaralı kayıtta ritim/form kodları vardır fakat bu görevin diagnostic
hedefi yoktur. Dışlama, kaydın klinik açıdan normal olduğu anlamına gelmez.

Ham veriler: `data/raw/ptb-xl/1.0.3/`.
Türetilen veriler: `data/processed/ptb-xl/1.0.3/ptbxl-superclass-presence-v1/`.
Burada tüm dahil edilen metadata için `labels.jsonl`, `label_audit.json` ve
`sample_labels.json` bulunur. Waveform yalnızca üç kayıtta indirilmiştir;
`labels.jsonl` içinde yer alması waveform'un indirilmiş olduğunu göstermez.
Ham/türetilen veriler mevcut `.gitignore` kapsamında Git dışında kalır.

## Provenance ve checksum

`provenance.json`; sürüm, DOI, lisans, politika, URL, UTC doğrulama zamanı,
bayt boyutu, SHA-256, örnek seçimi ve denetim sayılarını kaydeder.
Metadata, SCP tablosu, lisans, resmî örnek kod ve üç `.hea/.dat` çifti
olmak üzere 10 dosyanın SHA-256 değeri yayımlanan `SHA256SUMS.txt` ile karşılaştırıldı.
Manifest'in kendisi için yerel SHA-256 kaydedildi; bağımsız upstream imzası/hash'i
doğrulanmış değildir. Sonraki çalıştırmada kaydedilmiş manifest hash'i kullanılır.
Etiket schema'sı, üretici kod ve türetilen dosyalar ayrıca hash'lenir;
üretimde kullanılan Python sürümü de kaydedilir.

Tekrar çalıştırmak için proje kökünde:

```powershell
& .\.venv-3.12\Scripts\python.exe -m signai.ptbxl
```

İndirilen dosyalar hash ile yeniden kontrol edilir; eşleşmeyen veri tekrar
indirilir ve doğrulanmadan yayımlanmaz. Tam waveform arşivi indirilmez.
Tekrar çalıştırma doğrulama zamanını günceller; aynı girdilerin hedef/çıktı hash'i değişmemelidir.

## Kabul kontrolleri

- Python 3.12.13 ve 3.13.13: 28 test geçti (26 görev testi + 2 önceki smoke test).
- Tam üretim ikinci kez çalıştırıldı: schema ve üç türetilen çıktı hash'i aynı kaldı.
- Tüm 11 yerel kaynak dosyasının kayıtlı hash'i kontrol edildi;
  manifest dışındaki 10 dosya yayımlanan checksum'larla eşleşti.
- Bilinmeyen kod deneyi: `NORM` yanında `NEW_UNKNOWN_SCP` eklendiğinde
  `LabelValidationError: Unknown SCP code(s): NEW_UNKNOWN_SCP` alındı.
- Hasta/fold tutarlılığı, sabit etiket sırası, boş hedef dışlaması, tekrar eden
  superclass deduplikasyonu ve geçersiz güven değerleri kontrol edildi.

## Kapsam sınırları

Bu bir öğrenme/araştırma sınıflandırma görevidir. Bu aşamada klinik kullanım,
başka popülasyonlara genelleme veya model doğruluğu değerlendirilmedi.
Etiketler anotasyonlardan türetilir; class imbalance ve low-confidence
anotasyonların etkisi model değerlendirme aşamasında ayrıca ele alınacak.

## Resmî kaynaklar

- [PhysioNet PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- [Resmî superclass eşleme örneği](https://physionet.org/files/ptb-xl/1.0.3/example_physionet.py)
- [SCP kod tablosu](https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv)
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
