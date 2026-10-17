# DVC doğrulaması — 17 Ekim 2026

Ortam: Windows, Python 3.12.13, DVC 3.67.1 (`uv.lock` ile kilitli).
Git HEAD: `c9c8306c00903f2d079cd5dc4c7b9509b09c693b`.
Kullanıcıya bırakılan Git commit ve Git push yapılmadı. `dvc init`, kendi üç
başlangıç dosyasını Git index'ine ekledi; diğer değişiklikler ayrıca stage edilmeli.

## Sonuçlar

| Kontrol | Gerçek sonuç |
| --- | --- |
| Provenance'a karşı ham veri doğrulaması | 11 dosya, 14.978.789 byte, SHA-256 ve boyutlar eşit |
| DVC takibi | Ham klasör ve manifest için iki `.dvc` işaretçisi |
| Yerel remote'a `dvc push` | 13 nesne: 11 ham dosya + klasör indeksi + manifest |
| Bağımsız clone/cache | Başlangıçta ham veri, manifest payload'u ve cache yok |
| Yanlış remote deneyi | Boş dizine yönlendirilmiş `pull`, çıkış kodu 1; `Missing cache files` |
| Doğru remote ile `pull` | 13 nesne indirildi; ham dosyalar ve manifest oluşturuldu |
| İkinci klasör içerik kontrolü | 11 dosyanın SHA-256 ve boyutları provenance/manifest ile eşit |
| `dvc status` | `Data and pipelines are up to date.` |
| `dvc status --cloud` | `Cache and remote 'localstore' are in sync.` |
| Testler | 111 geçti; 5 yeni snapshot kontrolü dahil |
| Git geçmişi | Tüm yerel referanslardan erişilen 77 blob incelendi; 10 MiB üstü blob veya ham veri/manifest payload yolu bulunmadı |
| Git index politikası | Kontrol anında 73 dosya; ignore edilen veya 10 MiB üstü dosya yok |
| Remote/secret ayarı | Paylaşılan `.dvc/config` yalnızca analytics ayarını içeriyor; remote yolu ignore edilen `.dvc/config.local` içinde; yerel remote erişim anahtarı kullanmıyor |
| Ignore kontrolü | Sinyal, manifest payload'u, DVC cache ve yerel config/remote ignore ediliyor; `.dvc` işaretçileri ve veri `.gitignore` dosyası görünür |
| Diff biçimi | `git diff --check` geçti |

İlk pytest denemesinde sistemin mevcut `Temp/pytest-of-kadir` klasörüne erişim
engeli nedeniyle 39 fixture kurulum hatası oldu (72 test geçti). Proje içinde
benzersiz `--basetemp` kullanılarak tekrar çalıştırıldığında 111 testin tamamı geçti.
Bu geçici klasör hatası için ürün kodunda değişiklik yapılmadı.

## Deneyin sınırı

Bu deneme **HEAD clone + bekleyen DVC metadata kopyası** modunda yapıldı.
Yeni `.dvc` dosyaları henüz commit edilmediğinden Git bunları tek başına klonlayamaz.
Veri ve cache kopyalanmadı; dosyalar gerçek DVC remote'dan indirildi.
Commit sonrası `--working-tree-snapshot` bayrağı olmadan tekrar çalıştırılmalı.
Dolayısıyla yeni commit'in saf clone kontrolü kullanıcı commit'inden sonra tamamlanabilir.
Index kontrolü yeni dosyaların tümü stage edildikten sonra tekrar çalıştırılmalı.
Geçmiş taraması erişilebilir yerel referansları kapsar; uzak sunucunun silinmiş
referansları veya unreachable Git nesneleri hakkında sonuç vermez.

## Kanıtlar

Son deney klasörü (Git'e dahil edilmez):
`.dvc-lab/1b4b8ea8e4d14b7caebab532c912d5a8/`.
İçinde `result.json`, `broken-remote.log`, `pull.log` ve `clone/` bulunur.

Ham klasör DVC kimliği:
`34122c6f89c8cc6ba37b392bb94be39a.dir`.
Manifest DVC MD5 kimliği: `045460b1b3cadb1cffd27f2e61d236c9`.
Manifest SHA-256:
`7d5c9318340a6934ecacbfd46ba5209684fea72b3f6f22d506b8da844e1c0d54`.

```powershell
# Commit sonrası proje kökünde, rehberdeki uv ortam değişkenlerini ayarladıktan sonra:
uv run --locked python scripts/verify_dvc_roundtrip.py --remote "$PWD/.dvc-remote/ptbxl"
# Stage işleminden sonra:
uv run --locked python scripts/check_repository.py --history
```
