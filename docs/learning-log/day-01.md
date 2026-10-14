# Gün 1 — Proje sözleşmesi ve çalışma ortamı

Tarih: 14 Ekim 2026. Hedef: MLOps öğrenme projesinin aynı başlangıcını yeniden kurabilmek.

## Neyi, nasıl, neden yaptık?

### 1. Önce proje sözleşmesi

`docs/scope.md` ile öğrenme hedefini, bugünün çıktılarını ve açık kararları yazdık.
MLOps araç toplamak değildir: bir ML sistemini güvenilir biçimde geliştirme,
doğrulama, dağıtma ve işletme disiplinidir. Bugün yalnızca bunun ortam temelini kurduk.
Problem seçilmediği için doğruluk veya canlı servis maliyeti uydurmadık.

### 2. Git deposu ve yazar kimliği

`git init -b main` boş klasörü sürüm kontrollü bir depoya çevirdi.
Git commit, belirli bir andaki takip edilen dosyaların kaydıdır.
`git config --show-origin user.name`, `git config --show-origin user.email` ve
`git var GIT_AUTHOR_IDENT` ile geçerli yazar bilgisini doğruladık.
Kimlik global ayardan geliyor; bu, GitHub hesabına giriş yapıldığı anlamına gelmez.
Repo adı ile import adı aynı olmak zorunda değildir: dağıtım `signai-2.0`, import `signai`.

### 3. src/tests/docs yerleşimi

`src/signai` uygulama kodudur, `tests` kontrol kodudur, `docs` karar ve öğrenme kaydıdır.
`src` düzeninde paketi kurmadan import etmek beklenmez. Böylece proje klasörünün
Python arama yoluna otomatik eklenmesi bir paketleme hatasını gizleyemez.
Testlerde kaynak yolu eklemedik; ayrıca dış klasörden gerçek wheel importunu doğruladık.

### 4. pyproject, lockfile, sanal ortam

Üç farklı soruya cevap verirler:

| Parça | Cevapladığı soru | Örnek |
| --- | --- | --- |
| `pyproject.toml` | Bu proje ne istiyor, nasıl paketleniyor? | Python aralığı, pytest aralığı, build backend |
| `uv.lock` | Çözücü tam olarak hangi bağımlılıkları seçti? | pytest 8.4.2 ve alt bağımlılıkları, kaynaklar/hash'ler |
| `.venv-3.12` | Bu makinede bu proje için nereye kuruldu? | Ayrı interpreter ve site-packages |

`pytest>=8.3,<9` izin verilen aralıktır; `pytest==8.4.2` bu kilitte seçilen sürümdür.
Doğrudan bağımlılık bizim istediğimiz pytest; geçişli bağımlılık pytest'in ihtiyaç
duyduğu pluggy gibi paketlerdir. Lockfile ikisini de kaydeder.
Sanal ortam bağımlılıkları ayırır; container değildir ve işletim sistemini ayırmaz.

`uv lock` çözümü oluşturdu. `uv sync --locked` kilidin manifest ile tutarlı olduğunu
kontrol edip ortamı eşitledi; kilidi sessizce değiştirmesine izin vermedik.
`--frozen` kilidi değiştirmez ama güncelliğini kontrol etmeyi de atlar; bu yüzden
buradaki normal doğrulama komutunda `--locked` seçtik.
Bağımlılık değişikliklerinde manifest ve lockfile birlikte commit edilir.
Lockfile tek başına eğitimi birebir tekrarlanabilir yapmaz: veri, seed, driver,
framework, donanım ve bazı işlemlerin nondeterministik davranışı ayrıca önemlidir.

### 5. Smoke test

Smoke test temel sistemin ayağa kalktığını hızlıca kontrol eder.
İlk test kurulu dağıtımın metadata sürümü ile import edilen paketi karşılaştırır.
İkinci test yeni bir Python işlemi içinde `python -I -m signai` girişini çalıştırır.
`-I` çalışma klasörü ve PYTHONPATH gibi etkileri azaltır.
Bu testler model kalitesini, GPU eğitimini veya API davranışını ölçmez.

### 6. Secrets ve büyük dosyalar

`.gitignore` ile `.env`, anahtar dosyaları, veri/model dizinleri, sanal ortam ve
cache dosyalarını dışarıda bıraktık. `git check-ignore` ile örnek yolları kontrol ettik.
`.gitignore` daha önce takip edilmiş bir dosyayı Git geçmişinden kaldırmaz;
herhangi bir dosya adının içindeki sırrı da tespit etmez.
`scripts/check_repository.py` Git index'indeki ignore kapsamına giren veya
10 MiB üstü dosyaları reddeder. Bu bir içerik tabanlı secret scanner değildir.
Commit öncesi diff yine okunur. Veri ve model sürümlemesini sonraki aşamada
uygun araçla tasarlayacağız; büyük dosyayı Git'e koymak zorunda değiliz.

### 7. ADR

`docs/adr/0001-python-and-environment.md`: neden Python 3.12, uv ve src düzenini
seçtiğimizi, alternatifleri ve bedelini yazdık. ADR kararın hafızasıdır;
"Docker kur, API yaz" şeklinde yapılacaklar listesi değildir.

## Deney ve hata günlüğü

| Deneme | Gözlem | Çözüm / anlam |
| --- | --- | --- |
| Varsayılan uv cache | Kısıtlı ortamda yazma erişimi reddedildi | Cache ve Python indirme dizinlerini proje içindeki ignore edilmiş klasörlere aldık. |
| Python indirme | Kısıtlı ağda DNS hatası | Ağ erişimi olan çalıştırmayla resmi uv dağıtım kaynaklarından indirildi. |
| Python 3.12.13 | Kurulum, import, 2 test başarılı | Başlangıç sürümü doğrulandı. |
| Python 3.13.13 | İlk cache erişimi reddedildi; uygun erişimle kurulum ve 2 test başarılı | Bu erişim hatası Python uyumsuzluğu değildi. |
| Python 3.14.0 | requires-python aralığı nedeniyle reddedildi | Proje kuralı; 3.12/3.13 kullan. 3.14 desteği istenirse aralık ve bağımlılıklar ayrıca test edilir. |
| Temiz wheel kurulumu | Proje dışında site-packages içinden import başarılı | Kaynak klasörü yerine kurulu paketi doğruladık. |

## Defterine yaz

1. MLOps hedefim: modelin yanında veri, kod, ortam, dağıtım ve izleme yaşam döngüsünü yönetmek.
2. Manifest ihtiyacı bildirir; lockfile çözümü kaydeder; sanal ortam kurulan ortamdır.
3. Yeniden kurulum kanıtı: yeni venv + kilitli sync + proje dışından import + smoke test.
4. Her hatayı sürüm uyumsuzluğu sanma: ağ, izin, interpreter, dependency, uygulama diye ayır.
5. RAM sistem belleği; VRAM GPU belleğidir. Batch/model boyutu VRAM'i, veri hazırlama RAM'i zorlar.
6. Benim makinem: i7-12650H / ~16 GB RAM / RTX 4060 Laptop / ~8 GiB VRAM / ~383 GiB boş disk.
7. ADR: bağlam → karar → gerekçe → alternatif → bedel.
8. 33 saat henüz doğrulanmadı. Örnek dağılım: 5 gün × 5 saat + 1 gün × 8 saat;
   bir hafta gerçek saatleri kaydet, öğrenme/not/test süresini de say.
9. Bulut bütçesi henüz belli değil. Gelecekte eğitim saati, servis çalışma süresi,
   depolama ve ağ maliyetini ayrı hesapla; Colab ile sürekli servis aynı ihtiyaç değildir.

## Kendin dene (mevcut dosyaları değiştirmeden)

```powershell
.\scripts\bootstrap.ps1
.\scripts\bootstrap.ps1 -PythonVersion 3.13
git check-ignore .env data/example.csv models/example.pt
git status --short
```

Notuna beklediğin sonucu, gördüğün sonucu ve fark varsa nedenini yaz.
`uv.lock` içinde pytest ile onun bağımlılıklarını bul; "bunları ben tek tek mi seçtim?" sorusunu cevapla.

## Mülakatta anlatma örneği

"Önce kapsamı ve kabul koşullarını yazdım. Python paketini src düzeninde kurdum,
manifest ve lockfile'ı sürümledim. İki Python sürümünde smoke test çalıştırdım;
ayrıca temiz ortama wheel kurup kaynak klasörü dışında import ettim. Secrets ve
model çıktıları Git dışında. Lockfile'ın veri ve GPU sürücüsünü sabitlemediğini
biliyorum; sonraki aşamalarda bunların izlenebilirliğini ekleyeceğim."

Ezber yerine şu soruları kendi cümlelerinle cevapla:

- Test geçse bile neden modelin doğru çalıştığını söyleyemeyiz?
- Lockfile varken pyproject neden gerekli?
- 3.14 reddedildi diye pytest 3.14 ile uyumsuz diyebilir miyiz?
- Yerel Git deposu neden otomatik olarak GitHub yedeği değildir?

## Kaynaklar

- [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uv Python yönetimi](https://docs.astral.sh/uv/guides/install-python/)
- [Git kimlik ayarları](https://git-scm.com/docs/git-config)
- [Git ignore davranışı](https://git-scm.com/docs/gitignore)

Görevdeki "MOMENT" kaynağının tam bağlantısı verilmedi; herhangi bir içeriği
okunmuş veya doğrulanmış kabul edilmedi.
