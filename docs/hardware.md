# Yerel donanım envanteri

Ölçüm: 14 Ekim 2026. Disk boşluğu ölçüm anına aittir.

| Kaynak | Ölçülen değer | Pratik anlamı |
| --- | --- | --- |
| İşletim sistemi | Windows 11, build 26200, x86_64 | Kurulum kanıtı bugün bu işletim sistemi içindir. |
| CPU | Intel Core i7-12650H, 16 mantıksal işlemci | Veri hazırlama / CPU işleri için yerel kaynak. |
| RAM | 15.71 GiB kullanılabilir fiziksel toplam (~16 GB) | Sistem ve uygulamalar bunu paylaşır; tüm veri RAM'e yüklenmez. |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU | GPU eğitimi sonraki aşamada ayrıca doğrulanacak. |
| VRAM | 8188 MiB (~8 GiB) | Batch size ve model boyutu için önemli sınır. |
| GPU sürücüsü | 610.78 | Eğitim ortamı kaydının bir parçası. |
| nvidia-smi CUDA göstergesi | 13.3 | Sürücünün gösterdiği destek; kurulu CUDA Toolkit / PyTorch kanıtı değildir. |
| C: toplam disk | ~851.49 GiB | İşletim sistemi ile paylaşılır. |
| C: boş disk | ~383.19 GiB | Veri/cache/model çıktıları için sürekli izlenecek. |
| Başlangıç Python | 3.14.0 | Projenin seçtiği aralığın dışında. |
| uv | 0.11.14 | Ortam / bağımlılık yönetimi aracı. |
| Git | 2.51.1.windows.1 | Yerel sürüm kontrol aracı. |

RAM, Windows `GlobalMemoryStatusEx` API'si; disk, `shutil.disk_usage`; GPU,
`nvidia-smi`; CPU adı Windows kayıt defteri ile okundu.
`Get-CimInstance` kısıtlı çalıştırma ortamında erişim hatası verdi.
Fiziksel çekirdek sayısı ayrıca doğrulanmadı; mantıksal işlemci sayısıyla karıştırılmamalı.

## Kullanıcı notları

- Haftalık 33 saat: henüz onaylanmadı. Bir hafta gerçek çalışma süresini ölç.
- Bulut bütçesi: henüz kararlaştırılmadı; Colab düşünülüyor, ileride konuşulacak.
- İlk kaynak yaklaşımı: yerel çalışma; ücretli bulut yok.
