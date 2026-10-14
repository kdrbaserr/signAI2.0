# Proje sözleşmesi — 14 Ekim 2026

## Amaç

Kadir'in MLOps mühendisliğini uygulayarak kavraması ve iş mülakatlarında
kararlarını, hata ayıklama yaklaşımını ve sistemin sınırlarını açıklayabilmesi.
Başarı yalnızca modelin çalışması değil, aynı çalışma ortamının yeniden kurulması
ve her kararın kanıtıyla anlatılabilmesidir.

## Gün 1 kapsamı ve kabul koşulları

- Yerel `signai-2.0` Git deposu; `src/tests/docs` iskeleti.
- `pyproject.toml`, gerçek çözücüyle üretilmiş `uv.lock`, Python sürüm tercihi.
- Temiz sanal ortamda kurulabilen ve import edilebilen paket.
- Paket metadata ve modül girişini doğrulayan smoke test.
- Donanım envanteri, öğrenme günlüğü ve bir ADR.
- Git yazar kimliği kontrolü; secrets, veri ve model çıktılarının Git dışında tutulması.
- İkinci Python sürümünde kurulum deneyi; hata ve çözüm kaydı.

## Sonraki aşamaların yönü

Küçük bir ML problemi üzerinden veri doğrulama, veri/model sürümleme, eğitim,
deney takibi, değerlendirme, paketleme, servis, CI/CD ve izleme öğrenilecek.
Bu liste yön gösterir; bugün bu araçların tamamını kurma taahhüdü değildir.
Her aşamada bir çalışır çıktı, doğrulama kanıtı ve öğrenme notu üretilecek.

## Bugün kapsam dışında

Bulutta servis açmak, ücretli kaynak oluşturmak, büyük veri indirmek,
GPU eğitimi, Kubernetes ve otomatik dağıtım.

## Açık kararlar

| Konu | Durum |
| --- | --- |
| Veri / model problemi | Henüz seçilmedi; proje isminden işaret dili kapsamı çıkarılmadı. |
| Haftalık 33 saat | Plan hedefi; kullanıcı uygunluğu henüz doğrulanmadı. |
| Bulut / Colab bütçesi | Kullanıcı bütçeyi henüz tasarlamadı; ilerleyen aşamalarda görüşülecek. |
| Bugünkü harcama | Ücretli bulut kaynağı yok; yerel çalışma. |
| Uzak repo | Henüz yok; görünürlük ve hesap seçimi sonraki karar. |

## Çalışma anlaşması

Bir görev: neden → küçük uygulama → kontrol → hata/çözüm → öğrenme notu.
Tamamlanmayan kontroller tamamlanmış gibi yazılmaz. Model metriği, gecikme ve
maliyet hedefleri problem seçildikten sonra tanımlanır. Veri lisansı ve kişisel
veri koşulları veri seçimi sırasında incelenir.
