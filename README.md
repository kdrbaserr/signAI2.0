# SignAI 2.0

MLOps mühendisliğini uygulayarak öğrenmek için adım adım geliştirilen proje.
Gün 1 çıktısı: kurulabilir Python paketi, kilitli bağımlılıklar, smoke test ve öğrenme günlüğü.
Henüz model eğitimi veya canlı servis içermez.

## Kurulum (Windows / PowerShell)

Git ve [uv](https://docs.astral.sh/uv/getting-started/installation/) gerekir.
Proje kökünde çalıştır:

```powershell
.\scripts\bootstrap.ps1
# Aynı lockfile ile ikinci Python sürümü:
.\scripts\bootstrap.ps1 -PythonVersion 3.13
```

Betik Python'u gerekirse indirir; her sürüme ayrı sanal ortam kurar ve testleri çalıştırır.
PowerShell betik çalıştırmayı engelliyorsa sistem politikasını değiştirmeden şu komutları kullan:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.uv-python"
uv sync --locked --python 3.12
uv run --locked python -I -m signai
uv run --locked pytest
```

Bağımlılık değişikliği yaparken `uv add <paket>` veya `uv add --dev <paket>` kullan;
`pyproject.toml` ve `uv.lock` değişikliklerini birlikte incele. Normal kurulumda
`--locked` kilidin manifest ile tutarlı olmasını zorunlu tutar.

## Proje haritası

- `src/signai/`: kurulacak uygulama paketi; import adı `signai`.
- `tests/`: ortamın ve paket girişinin smoke testleri.
- `docs/scope.md`: amaç, kapsam, başarı koşulları ve açık kararlar.
- `docs/hardware.md`: ölçülen yerel kaynaklar.
- `docs/learning-log/day-01.md`: açıklamalar, deneyler ve not soruları.
- `docs/adr/0001-python-and-environment.md`: ortam tercihinin gerekçesi ve bedeli.
- `docs/verification-day-01.md`: gerçek doğrulama sonuçları.

Repo/paket adı `signai-2.0`; mevcut çalışma klasörü `signAI2.0` korunur.
Yerel Git deposu kullanılır. Uzak GitHub deposu henüz tanımlı değildir.

## DVC ile ham veri sürümleme

PTB-XL 1.0.3 metadata + üç örnek EKG ve ham veri manifesti DVC ile izlenir.
Kurulum, clone/pull deneyi, hata ayrımı ve öğrenme notları:
[DVC çalışma rehberi](docs/learning-log/day-02-dvc.md).
Gerçek çalışma sonuçları: [DVC doğrulaması](docs/verification-dvc.md).

## Commit öncesi

```powershell
git add .
python scripts/check_repository.py --history
git diff --cached
```

Kontrol betiği index'teki ignore kapsamındaki dosyaları ve 10 MiB üstü dosyaları
reddeder. `--history` tüm yerel referanslardan erişilen geçmiş nesneleri de denetler;
10 MiB üstü blob ve ham veri payload'u arar. Manuel çalıştırılır; otomatik hook değildir ve içerik tabanlı secret
taraması yapmaz. Diff'i okuyup ardından commit yap.
