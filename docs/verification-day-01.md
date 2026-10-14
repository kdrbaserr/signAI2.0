# Gün 1 doğrulama kanıtı

14 Ekim 2026, Windows 11. İlk kurulumlar boş sanal ortamlara yapıldı.

| Kontrol | Sonuç |
| --- | --- |
| `uv lock --python 3.12` | Gerçek çözücü 7 paket çözdü; `uv.lock` üretildi. |
| `scripts/bootstrap.ps1` | CPython 3.12.13, pytest 8.4.2; 2 test geçti. |
| `scripts/bootstrap.ps1 -PythonVersion 3.13` | CPython 3.13.13, aynı lockfile; 2 test geçti. |
| `uv sync --locked --no-editable --python 3.12` | Yeni `.venv-clean-check` ortamına wheel kuruldu. |
| Proje dışından `python -I -c "import signai"` | 0.1.0; `.venv-clean-check/Lib/site-packages/signai/__init__.py` içinden import. |
| Proje dışından `python -I -m signai` | `signai-2.0 0.1.0: environment ready`. |
| Temiz wheel ortamında `python -I -m pytest` | 2 test geçti. |
| Python 3.14.0 ile locked sync | Beklenen reddetme: `>=3.12,<3.14` gereksinimi dışında. |
| Git yazar bilgisi | Global config'ten Kadir; e-posta tanımlı ve `GIT_AUTHOR_IDENT` ile doğrulandı. |
| `git check-ignore` | `.env`, veri/model örnekleri, uv cache/Python ve venv yolları dışarıda. |
| `python scripts/check_repository.py` | 16 index dosyası; ignore kapsamındaki veya 10 MiB üstü dosya yok. |
| `git diff --cached --check` | Whitespace hatası yok. |

Sınırlar: Windows üzerinde doğrulandı; Linux/macOS, CI, GPU eğitimi ve canlı servis
henüz test edilmedi. Lockfile cache'in sıfır olmasını gerektirmez; temiz venv testi
ile tüm indirmelerin çevrimdışı veya cache'siz tekrarlanması farklı kontrollerdir.

## Temiz kurulum deneyini yeniden çalıştır

Var olan ortamı silmek yerine yeni, kullanılmamış bir ortam adı seç:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.uv-python"
$env:UV_PROJECT_ENVIRONMENT = "$PWD\.venv-fresh-$([guid]::NewGuid().ToString('N'))"
uv sync --locked --no-editable --python 3.12
$verifiedPython = "$env:UV_PROJECT_ENVIRONMENT\Scripts\python.exe"
& $verifiedPython -I -m pytest
Push-Location $env:TEMP
try {
    & $verifiedPython -I -c "import signai; print(signai.__version__); print(signai.__file__)"
    & $verifiedPython -I -m signai
} finally {
    Pop-Location
}
```
