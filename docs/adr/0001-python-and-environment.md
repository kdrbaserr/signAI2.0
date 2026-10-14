# ADR 0001 — Python ve kurulum yönetimi

Tarih: 14 Ekim 2026. Durum: Gün 1 için kabul edildi.

## Bağlam

Makinede Python 3.14.0 bulunuyor. Proje bir MLOps öğrenme çalışması;
ilk gün model/framework seçilmedi. Aynı kurulumu başka ortamda tekrar edebilmek
ve yeni bağımlılıkların çözümünü kontrol edebilmek gerekiyor.

## Karar

- Başlangıç Python sürümü 3.12; uyumluluk deneyi 3.13.
- İlan edilen destek aralığı `>=3.12,<3.14`.
- uv ile yönetilen sanal ortamlar ve Git'e giren `uv.lock`.
- `src` paket yerleşimi ve Hatchling build backend (1.27.0).
- Runtime bağımlılığı yok; pytest yalnızca geliştirme grubunda.
- Normal yeniden kurulum `uv sync --locked` ile yapılır.

## Gerekçe ve bedel

3.12 konservatif bir başlangıç tercihidir; 3.14'ün tüm ML araçlarıyla uyumsuz
olduğu iddia edilmez. Aralık bir proje politikasıdır. ML framework seçilince
sürüm desteği resmi kaynaklarla yeniden değerlendirilecek.

uv hızlı çözümleme ve lockfile sağlar; ek bir araç kurulumu ve onun sürümünü
kaydetme sorumluluğu getirir. Lockfile bağımlılık sürümlerini, kaynaklarını ve
hash'lerini kaydeder; işletim sistemi, GPU sürücüsü, veri, rastgelelik ve donanımı
tek başına sabitlemez. Build backend ayrıca tam sürüme sabitlenmiştir;
normal uv lockfile'ın build izolasyonuna tümüyle hükmettiği varsayılmaz.

`src` düzeni paketin gerçekten kurulmasını gerektirir. Bu küçük bir ek adımın
karşılığında çalışma klasörünün import hatalarını gizlemesini azaltır.

## Alternatifler

- pip + venv: geçerli seçenek; bağımlılık kilidi için ayrıca süreç gerekir.
- Conda: Python dışı bağımlılıklarda yararlı olabilir; bugün gerekmiyor.
- Sistem Python'u: hızlı başlangıç; proje bağımlılıklarını birbirine karıştırabilir.

ADR yapılacaklar listesi değildir: bağlam, karar, gerekçe ve sonuçları kaydeder.
Karar değişirse yeni bir ADR eski kararı geçersiz kılabilir.
