# CV-SAR 원시데이터 신경망 압축 — 논문 재구현

**논문**: R. Mohammadi Asiyabi, M. Datcu, A. Anghel, A. Focsa, M. Martone, P. Rizzoli, E. Imbembo,
"Complex-Valued Autoencoder-Based Neural Data Compression for SAR Raw Data",
*IEEE Journal of Selected Topics in Signal Processing*, vol. 19, no. 3, pp. 572–582, 2025.
DOI: [10.1109/JSTSP.2025.3558651](https://doi.org/10.1109/JSTSP.2025.3558651)

논문의 전체 실험 흐름(Fig. 3)을 Python(PyTorch)으로 구현한 코드입니다.

```
Sentinel-1 L0(.dat, FDBAQ 압축) ─▶ L0 디코딩 ─▶ 적응(adaptation, 참고문헌 [20])
      ─▶ 복소수 오토인코더 학습 (λ 별, RD 손실)
      ─▶ CAE / BAQ / JPEG2000 압축·복원 ─▶ Range-Doppler 포커싱(SLC)
      ─▶ SQNR · 위상오차 표준편차 · 복소 코히어런스 ─▶ Fig. 4–9, Table I–III
```

---

## 1. 논문 구성 요소와 코드의 대응

| 논문 | 내용 | 코드 |
|---|---|---|
| Fig. 2 | CV 인코더 `Conv(1→192)–GDN–Conv(192→192)–GDN–Conv(192→192)–GDN–Conv(192→320)`, 5×5 커널, stride 2, zero padding 2 | `cvsarc/model.py` `CVSARCompressor.g_a` |
| Fig. 2 | CV 디코더 `TConv(320→192)–IGDN–…–TConv(192→1)`, padding 2, output padding 1, 출력층 활성화 없음 | `CVSARCompressor.g_s` |
| Sec. II-D-2 | CV 합성곱 `(W_r+jW_i)*(x_r+jx_i)`, CV-GDN/IGDN | `cvsarc/complex_layers.py` |
| Eq. (8) | Wirtinger 미분 기반 CV 역전파 | PyTorch 복소 autograd (별도 구현 불필요) |
| Sec. II-D-2 | 양자화: 학습 시 균등잡음 `U(-½,½)`, 추론 시 반올림 | `cvsarc/entropy_models.py` |
| Eq. (2) | 비율 항 `L_R = E[-log2 p(s)]`, 분해형(factorized) 엔트로피 모델 | `EntropyBottleneck` |
| Fig. 2 AE/AD | 산술 부호화기 / 복호화기 (실제 비트스트림 생성) | `cvsarc/entropy_coder.py`, `cvsarc/codec.py` |
| Eq. (1), (7) | `L_RD = L_R + λ·L_D`, `L_D = MSE` | `cvsarc/model.py` `RateDistortionLoss` |
| Sec. II-A | Sentinel-1 L0 (FDBAQ/BAQ/bypass) 디코딩, 궤도 정보 | `cvsarc/sentinel1_l0.py`, `cvsarc/s1_tables.py` |
| Sec. II-A, [20] | FDBAQ 복호 데이터의 적응(양자화 간격을 균등 잡음으로 채움) | `cvsarc/adaptation.py` |
| Sec. II-B | BAQ 기준 방법 | `cvsarc/baq.py` |
| Sec. II-C | JPEG2000 기준 방법 (실수부·허수부 별도 압축) | `cvsarc/jpeg2000_codec.py` |
| Fig. 3 | 원본·복원 원시데이터 → SLC 포커싱 | `cvsarc/focusing.py` (Range-Doppler) |
| Sec. III-A | SQNR, 위상오차 std, 복소 코히어런스 | `cvsarc/metrics.py` |
| Sec. III-B | 2·3·4 bpp 보간 표(Table I–III), 이득, 15/20 dB 목표에서의 비트율 절감 | `metrics.gain_tables`, `metrics.bitrate_at_quality` |
| Sec. III-C | ERS-1 일반화 실험 | `cvsarc/ers_reader.py` |
| Fig. 4–9 | RD 곡선, SLC 비교, 크기차 히스토그램 | `cvsarc/plotting.py`, `cvsarc/evaluation.py` |
| (검증용) | 합성 stripmap 원시데이터 + Sentinel-1 ISP 패킷 에뮬레이션 | `cvsarc/simulation.py` |

---

## 2. 설치

Python 3.10 이상에서 동작합니다(3.13에서 테스트).

```powershell
cd cv_sar_compression
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Windows에서 위 명령은 CPU용 torch를 설치합니다. NVIDIA GPU로 학습하려면 아래 2.1절을 따르세요.
- `numba`는 선택 사항이지만 강력히 권장합니다. 없으면 L0 디코더와 산술 부호화기가 순수 Python으로 동작해 매우 느립니다.

설치 확인:

```powershell
python -m pytest -q tests
```

### 2.1 NVIDIA GPU로 학습할 때

논문 크기 모델(N=192, M=320, 256×256 패치)은 CPU로는 사실상 학습할 수 없습니다. 4스레드 노트북 CPU에서 반복 1회에 약 39초, 1,000회에 약 11시간이 걸렸습니다.

GPU를 쓰려면 가상환경을 켠 뒤, `requirements.txt`보다 CUDA용 torch를 먼저 설치하세요. CUDA용 torch는 약 2 GB를 내려받습니다.
설치 명령은 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서도 확인할 수 있습니다.

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- 마지막 명령이 `True`와 GPU 이름을 출력해야 합니다. `False`가 나오면 NVIDIA 드라이버를 최신 버전으로 업데이트하세요.
- CPU용 torch가 이미 설치되어 있으면 첫 줄 끝에 `--force-reinstall --no-deps`를 붙여 교체합니다.
- torch 2.15의 CUDA 빌드는 Turing 이후 GPU(GTX 16·RTX 20 시리즈 이상)만 지원합니다. GTX 10 시리즈나 MX250 같은 이전 세대 GPU에서는 CPU로 학습됩니다.
- 장치는 자동으로 고릅니다. 학습 로그에 `device=cuda`가 보이면 GPU로 학습하고 있는 것입니다.
- 학습 1회에 필요한 메모리를 CPU에서 재 보니 배치 8에서 약 4.2 GB, 배치 4에서 약 2.4 GB였습니다.
  6 GB GPU(예: RTX 4050 노트북)에서는 배치 8이 한계에 가깝습니다. 첫 반복에서 `CUDA out of memory` 오류가 나면 `--batch-size 4`로 줄이세요.
- 기본 설정은 λ당 최대 100 epoch이고, 배치 8이면 1 epoch이 1,000회 반복입니다. 첫 epoch에 걸린 시간을 보고 `--max-iters`로 학습량을 정하세요.
- 학습이 중간에 끊기면 같은 명령에 `--resume`을 붙여 이어서 합니다.

---

## 3. 빠른 시작: 합성 데이터로 전체 파이프라인 실행

실제 데이터를 내려받기 전에 전체 흐름을 확인하는 데모입니다.

```powershell
python scripts/run_demo.py --out demo_output
```

데모는 다음 순서로 진행합니다.

1. 합성 장면(도시, 혼합, 농촌, 해안)을 만듭니다.
2. FDBAQ와 유사한 온보드 압축을 적용하고, 테스트 장면은 실제 Sentinel-1 ISP 패킷(`.dat`)으로 기록합니다.
3. L0 디코더로 `.dat`를 다시 읽고 적응 처리를 합니다.
4. 폭을 줄인 모델(N=64, M=320, 64×64 패치, Haar 초기화)을 λ 16, 64, 256으로 600 반복씩 학습합니다.
5. CAE, BAQ, JPEG2000을 비교하고 그림과 표를 `demo_output/results/`에 만듭니다.

`quick` 프리셋은 파이프라인 검증용이며, 논문의 수치를 재현하지는 않습니다.
논문 크기 모델은 `--preset standard`로 실행할 수 있습니다(GPU 권장).

### 데모 결과 예시

`examples/demo_quick/`에 데모 결과(그림, 표)가 들어 있습니다. 4스레드 노트북 CPU에서 학습과 평가에 약 11분 걸렸습니다.
합성 테스트 장면에서 3 bpp로 보간한 SLC 지표는 다음과 같습니다.

| 방법 | SQNR (dB) | 위상오차 std (°) | 코히어런스 |
|---|---|---|---|
| BAQ | 15.79 | 29.55 | 94.83% |
| JPEG2000 | 18.97 | 24.73 | 96.70% |
| CAE (축소 모델, λ당 600 반복) | 18.00 | 24.81 | 96.33% |

- 짧게 학습한 축소 모델도 BAQ보다 SQNR이 2.2 dB 높고, 목표 SQNR 15 dB에서 비트율을 15% 줄였습니다.
- JPEG2000보다는 아직 1 dB 낮습니다. 논문처럼 JPEG2000을 1~2 dB 앞서려면 논문 크기 모델을 실제 데이터로 충분히 학습해야 합니다.
- L0 디코더 출력은 에뮬레이션한 FDBAQ 재구성과 비트 단위로 같았습니다. 적응 후 I/Q 히스토그램과 비압축 데이터 사이의 거리(total variation)는 0.224에서 0.015로 줄었습니다.
- 논문의 "적응 없는 BAQ 3 bit" 이상치(주황색 ×)는 실제 FDBAQ 격자에 따라 생기는 현상이라, 에뮬레이션한 합성 데이터에서는 거의 나타나지 않습니다(16.11 dB 대 15.97 dB).

---

## 4. 실제 데이터로 논문 재현

### 4.1 데이터 준비

| 용도 | 논문의 데이터 | 비고 |
|---|---|---|
| 학습 | Sentinel-1 StripMap(SM) L0, HH 편파: Chicago, Houston(미국), São Paulo(브라질) | 장면 3개 |
| 테스트 | Sentinel-1 SM L0, Cape Verde(São Filipe 시가지·공항·나지·식생·해양) 부분 영역 | 별도 장면 |
| 일반화 | ERS-1 원시데이터, La Coruña(스페인) 1992년 12월 유류유출, VV 편파 | FDBAQ 미적용 → 적응 불필요 |

Sentinel-1 L0 제품은 [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/)에서 받을 수 있습니다.
검색 조건은 Product type `RAW`, Sensor mode `SM`입니다.
`.SAFE` 폴더 안의 `s1?-s?-raw-s-hh-*.dat`가 측정 파일입니다.

### 4.2 L0 디코딩

```powershell
# 에코 획득 구간(chunk) 확인
python scripts/decode_s1_l0.py --input S1A_S3_RAW__0SDH_xxx.SAFE --pol hh --list

# 가장 큰 에코 구간에서 일부만 디코딩 (예: 8192 라인)
python scripts/decode_s1_l0.py --input S1A_S3_RAW__0SDH_xxx.SAFE --pol hh --out data/chicago_l0 --line-start 2000 --max-lines 8192
```

출력 폴더에는 `raw.npy`(complex64, 방위×거리), `metadata.json`(포커싱 파라미터, 궤도), `side_info.npz`(블록별 BRC/THIDX)가 생깁니다.

### 4.3 적응(adaptation, 참고문헌 [20])

```powershell
python scripts/adapt_raw.py --input data/chicago_l0 --out data/chicago_adapted
```

각 샘플을 FDBAQ 결정 셀 안에서 균등 분포로 다시 뽑습니다. 셀은 블록의 (BRC, THIDX)로 복원한 재구성 레벨의 중점으로 정합니다.
그 다음 정수 ADC 격자로 반올림하여 "준균등(quasi-uniform)" 양자화 데이터를 만듭니다.
학습 장면 3개와 테스트 장면 모두에 적용합니다.

### 4.4 학습 (λ 별로 모델 1개씩)

```powershell
python scripts/train.py --train data/chicago_adapted data/houston_adapted data/saopaulo_adapted --lmbda 16 32 64 128 256 512 --out runs
```

- 기본 설정은 `configs/default.yaml`에 있습니다.
- 장치는 자동으로 고릅니다. CUDA를 쓸 수 있으면 GPU, 아니면 CPU에서 학습합니다.
- 각 λ마다 `runs/lambda_X/best.pt`가 생기고, 여기에 산술 부호화용 CDF 표가 함께 저장됩니다.
- 대략 `SQNR(dB) ≈ 10·log10(2·ln2·λ)` 관계가 있어 λ로 동작점을 고를 수 있습니다.

학습 관련 참고 사항입니다.

- **수렴 속도**: 기본값은 일반적인 무작위 초기화입니다. SAR 원시데이터는 거의 백색 잡음과 같아서, 네트워크가 16배 다운샘플링을 거치는 거의 항등인 변환을 처음부터 배워야 합니다. 그래서 수렴이 느리므로 GPU에서 충분히 오래 학습하세요.
- **빠른 수렴 옵션 (논문에 없음)**: `--init haar --entropy-lr-mult 50 --lr 1e-4`를 쓰면 인코더·디코더가 정확히 역변환 관계인 Haar 웨이블릿 패킷(16×16 블록의 Walsh-Hadamard 변환)으로 시작합니다. 잠재 크기는 λ에 맞춰 정해지므로 학습이 스칼라 양자화 수준의 동작점에서 출발합니다.
  - 엔트로피 모델만 빠르게 적응시키고 변환의 학습률은 1e-4로 두세요. 5e-4처럼 높이면 Adam이 0으로 초기화된 수많은 가중치를 잡음처럼 움직여 채널 간 간섭이 생기고, 높은 λ에서 SQNR이 20 dB 근처로 떨어졌습니다.
  - `--init polyphase`(순수 space-to-depth)도 있지만 권장하지 않습니다. 모든 잠재 채널의 분산이 같아 부호화 이득이 0 dB인 안장점에서 출발하기 때문에, 엔트로피 부호화된 스칼라 양자화 수준(λ=64에서 손실 ≈ 4.21)에 머물렀습니다. Haar는 처음부터 채널 분산이 달라 합성 데이터에서 1.35 dB의 부호화 이득이 있습니다.
- **CPU 실험 결과 (N=64, M=320, 64×64 패치, λ=64)**: 300 반복 후 검증 손실이 polyphase 시작은 4.25(3.38 bpp, 18.65 dB), Haar 시작은 4.04(3.19 bpp, 18.76 dB)였습니다. Haar 쪽은 스칼라 양자화 한계(4.21)보다 낮습니다. 무작위 초기화(N=48, λ=16)는 500 반복 후에도 원시 SQNR 6.4 dB에 머물렀습니다.
- **모델을 줄일 때 주의**: 잠재 채널 수 M은 줄이지 마세요. 16×16 블록은 실수 512개이고 잠재는 실수 2M개이므로, M < 256이면 정보가 구조적으로 손실됩니다. 논문은 M=320을 씁니다. N도 64 이상이어야 3번째 층에서 병목이 생기지 않습니다. 시험 삼아 N=32, M=48로 학습했을 때 SQNR이 1~2 dB에 머물렀습니다.
- **다른 센서로 미세조정**: `--init-from runs/lambda_64/best.pt`로 기존 모델에서 시작할 수 있습니다. 논문 Sec. III-C에서 언급한 ERS-1 같은 경우입니다.
- **λ 사이 이어 학습**: `--warm-start`를 주면 각 λ 모델을 직전 λ 모델에서 이어 학습합니다.

### 4.5 평가 (Fig. 4–6, 8, 9 / Table I–III)

```powershell
python scripts/evaluate.py --test data/capeverde_adapted --cae "runs/lambda_*/best.pt" --nonadapted data/capeverde_l0 --out results/capeverde
```

`results/capeverde/`에 다음 결과가 생깁니다.

| 파일 | 내용 |
|---|---|
| `fig4_sqnr.png`, `fig5_phase_error.png`, `fig6_coherence.png` | 비트율-품질 곡선 |
| `fig8_slc_examples.png` | 원본·CAE·BAQ·JPEG2000 SLC 비교 |
| `fig9_magnitude_difference_hist.png` | 정규화 크기차 히스토그램 |
| `tables.md` | Table I–III, 15/20 dB 목표에서의 비트율 절감 |
| `results.csv`, `results.json` | 모든 동작점의 수치 |

`--nonadapted`로 적응 전 데이터를 주면, 논문의 주황색 ×에 해당하는 "적응 없는 BAQ 3 bit" 점이 그림에 추가됩니다.

### 4.6 ERS-1 일반화 실험 (Fig. 7)

```powershell
# configs/ers1_params_template.yaml 의 prf, near_range, effective_velocity를 장면 값으로 채운 뒤
python scripts/read_ers_ceos.py --dat DAT_01.001 --params configs/ers1_params_template.yaml --out data/ers1_lacoruna --max-lines 8192
python scripts/evaluate.py --test data/ers1_lacoruna --cae "runs/lambda_*/best.pt" --out results/ers1
```

### 4.7 실제 비트스트림 압축·복원

```powershell
python scripts/compress.py encode --ckpt runs/lambda_64/best.pt --input data/capeverde_adapted --output cv.cvsar
python scripts/compress.py decode --ckpt runs/lambda_64/best.pt --input cv.cvsar --output data/capeverde_rec --meta-from data/capeverde_adapted
python scripts/focus.py --input data/capeverde_rec --out results/slc_rec --look 4 1
```

---

## 5. 논문에 명시되지 않아 가정한 부분

논문에 값이 없는 항목은 아래처럼 정했습니다. 모두 `configs/default.yaml` 또는 명령행 옵션으로 바꿀 수 있습니다.

- **λ 값과 학습 설정**: λ ∈ {16, 32, 64, 128, 256, 512}, Adam lr 1e-4, 보조(quantile) lr 1e-3, 배치 8, ReduceLROnPlateau. CompressAI 관례를 따랐습니다.
- **초기화**: 기본은 무작위 초기화입니다. 복소 Rayleigh 크기와 균등 위상을 쓰는 Trabelsi 방식입니다. `haar`·`polyphase` 초기화, `entropy_lr_mult`, `warmup_iters`는 수렴을 빠르게 하려고 추가한 선택 기능이며 논문에는 없습니다. 모델 구조와 손실 함수는 바꾸지 않습니다.
- **엔트로피 모델**: Fig. 2에 하이퍼프라이어가 없으므로 Ballé의 분해형 사전분포를 사용했습니다. 복소 잠재 채널 320개의 실수부·허수부에 각각 밀도를 두어 640개 채널로 모델링합니다.
- **CV-GDN 정의**: `y_i = x_i / sqrt(β_i + Σ_j γ_ij |x_j|²)` 형태로 위상을 보존합니다. 실수부·허수부에 따로 적용하는 방식은 `gdn_mode: split`으로 고를 수 있습니다.
- **입력 정규화**: 패치(타일)마다 성분 RMS로 나누고, 그 값(float32)을 부가 정보로 비트스트림에 넣습니다. 비용은 0.00024 bpp입니다.
- **비트율 단위 bpp**: 실수 표본(I 또는 Q) 1개당 비트입니다. 논문 그림에서 N비트 BAQ가 N bpp에 놓이는 것과 같은 정의입니다.
- **BAQ**: 블록은 거리 방향 128 샘플, Lloyd-Max 가우시안 양자화기, 블록 σ는 8비트 로그 코드북입니다. 부가 정보까지 포함해 비트율을 셉니다.
- **JPEG2000**: OpenJPEG(Pillow), 9/7 비가역 웨이블릿을 씁니다. 실수부·허수부를 16비트로 따로 압축하고, 목표 비트율은 이분 탐색으로 맞춥니다.
- **적응 [20]**: "적절한 진폭의 균등 잡음으로 양자화 레벨 사이 간격을 채운다"를 셀 단위 균등 재추출과 정수 반올림으로 구현했습니다.
- **포커싱**: 표준 RDA입니다. 거리·방위 가중은 generalized Hamming α=0.75, 처리 도플러 대역은 0.8·PRF, 도플러 중심은 ACCC로 추정합니다.
- **지표**: 위상오차는 `arg(s·ŝ*)`의 표준편차(도)이고, 코히어런스는 5×5 박스카 국소 추정의 평균입니다. 전역 추정값도 함께 기록합니다.
- **FDBAQ 인코더(시뮬레이션 전용)**: 실제 온보드 결정 규칙이 공개되어 있지 않아 에뮬레이션했습니다. 디코더와 재구성 표는 ESA 명세(S1-IF-ASD-PL-0007)와 일치하며, 공개 디코더 `sentinel1decoder`의 표와 대조했습니다.

---

## 6. 논문 결과 (참고)

| bpp | SQNR (dB) CAE / BAQ / JP2 | 위상오차 std (°) CAE / BAQ / JP2 | 코히어런스 CAE / BAQ / JP2 |
|---|---|---|---|
| 2 | 15.9 / 11.6 / 14.1 | 28.2 / 39.3 / 33.8 | 95.4% / 90.2% / 92.4% |
| 3 | 20.3 / 14.7 / 19.0 | 20.4 / 31.9 / 23.4 | 97.4% / 94.0% / 96.3% |
| 4 | 24.8 / 20.0 / 24.0 | 12.5 / 18.8 / 13.0 | 99.3% / 98.3% / 99.1% |

목표 SQNR 15 dB에서 CAE는 1.8 bpp, BAQ는 3.1 bpp가 필요해 약 42% 절감입니다. 20 dB에서는 2.9 bpp 대 4 bpp로 약 27% 절감입니다.

---

## 7. 폴더 구조

```
cv_sar_compression/
├── cvsarc/                 # 라이브러리
│   ├── complex_layers.py   # CV Conv / TConv / GDN / IGDN
│   ├── entropy_models.py   # 분해형 엔트로피 모델
│   ├── entropy_coder.py    # 산술 부호화기 (numba 가속)
│   ├── model.py            # Fig. 2 모델 + RD 손실
│   ├── codec.py            # 타일 단위 압축 컨테이너(.cvsar)
│   ├── training.py         # 학습 루프
│   ├── baq.py, jpeg2000_codec.py
│   ├── sentinel1_l0.py, s1_tables.py, adaptation.py, ers_reader.py
│   ├── focusing.py, metrics.py, evaluation.py, plotting.py
│   ├── simulation.py, datasets.py, config.py, utils.py
├── scripts/                # 실행 스크립트 (decode, adapt, train, evaluate, compress, focus, demo, ...)
├── configs/                # default.yaml, ers1_params_template.yaml
├── examples/demo_quick/    # 합성 데이터 데모 결과 (그림, tables.md, results.csv)
└── tests/                  # 단위·통합 테스트 (test_all.py, test_l0_modes.py)
```

## 8. 제한 사항

- ERS 원시데이터는 CEOS 형식만 읽습니다. Envisat 형식(`SAR_IM__0P`)은 별도로 변환해야 합니다.
- `sentinel1decoder` 백엔드(`--backend sentinel1decoder`)는 실험적이며, 이 백엔드에서는 BRC/THIDX 부가 정보가 없어 적응이 `generic` 모드로 동작합니다.
- 실제 Sentinel-1 파일로는 테스트하지 못했습니다. 디코더는 명세 기반 ISP 패킷을 비트 단위로 왕복 검증했고, 표는 공개 디코더와 대조했습니다.
