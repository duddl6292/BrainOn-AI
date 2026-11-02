# BrainOn — Multimodal Stroke Imaging AI

**CT·MRI·MRA 의료영상에서 뇌졸중 관련 병변을 분할하고, 임상 의사결정 지원 시스템(CDSS)에 연계하기 위한 AI 모델 개발 프로젝트**

BrainOn은 출혈성 병변(CT), 허혈성 병변(MRI DWI·ADC), 뇌동맥류(MRA)를 분석하는 의료영상 AI 프로젝트입니다.

본 저장소는 데이터 품질 검증부터 모델 학습·평가, 실험 이력 관리, 모델 릴리스까지의 개발 과정을 담고 있습니다.

| 항목 | 내용 |
| --- | --- |
| 프로젝트 | BrainOn — Stroke Clinical Decision Support System |
| 참여 형태 | 3인 팀 프로젝트 |
| 담당 역할 | 팀 리드 · AI 모델 개발 및 성능 평가 · 모델 연계 조정 |
| 주요 분야 | Medical Image Segmentation, Lesion Detection, Model Evaluation |
| 기술 | Python, PyTorch, nnU-Net, MONAI, MLflow, Google Cloud Storage |

> **연구·교육용 프로젝트**  
> 본 시스템은 의료인의 판단을 보조하기 위한 연구 결과이며, 독립적인 임상 진단 또는 치료 결정에 사용하도록 검증된 의료기기가 아닙니다.

---

## About This Repository

이 저장소는 3인 팀 프로젝트 **BrainOn**의 AI 모델 저장소를 바탕으로, **김호영의 모델 개발·실험·성능 평가 기여를 정리한 개인 포트폴리오**입니다.

- **팀 원본 저장소:** [brainCDSS/model](https://github.com/brainCDSS/model)
- **개인 포트폴리오:** [duddl6292/BrainOn-AI](https://github.com/duddl6292/BrainOn-AI)
- **개인 기여 상세:** [My Contributions](#3-my-contributions)

기존 개발 이력과 팀 공동 작업의 출처를 유지하며, 아래 성과와 개인 기여 범위를 함께 기록합니다.

## 1. Key Results

아래 수치는 각 실험의 **Validation 결과**입니다.

| Modality | Task | Model / Experiment | Macro Dice ↑ | 주요 병변 단위 지표 |
| --- | --- | --- | ---: | --- |
| CT | Intracranial Hemorrhage Segmentation | H-T01-E300 | 0.6264 | Lesion Detection Rate 63.18% |
| MRI | Ischemic Stroke Lesion Segmentation | I-E01 | 0.8095 | Lesion Recall 69.32%, Lesion F1 0.7474 |
| MRA | Intracranial Aneurysm Segmentation | A-B04 | 0.5737 | Lesion Detection Rate 82.50% (33/40) |

- **Dice**는 분할 영역의 겹침 정도를 나타냅니다.
- **Lesion Detection Rate**는 개별 병변 탐지 비율을 나타냅니다.
- **FP/case 또는 FP/scan**은 평가 데이터 한 건당 오탐으로 분류된 예측 병변 수입니다.
- 병변 매칭·평균 방식은 실험별로 다를 수 있으므로, 서로 다른 Modality의 지표를 직접 비교하지 않습니다.
- 특히 CT의 표기 탐지율은 정답 병변과 예측 영역이 **1 voxel 이상 겹치는 경우**를 기준으로 합니다.
- 평가 조건과 수치는 아래에 연결된 각 실험의 원본 기록에서 확인할 수 있습니다.

---

## 2. Model Development & Selection

### 2.1 CT — Intracranial Hemorrhage Segmentation

**최종 선정 모델: `H-T01-E300`**

뇌출혈 병변 분할을 위해 nnU-Net 기반 모델을 학습하고, Tversky Loss 설정과 학습 전략을 비교했습니다.

특히 `H-T01-E300`과 `H-T02-E300`을 동일한 **Validation 40건**에서 비교해 최종 모델을 선정했습니다.

| Validation Metric | H-T01-E300 | H-T02-E300 |
| --- | ---: | ---: |
| Macro Dice ↑ | 0.6264 | 0.6262 |
| Lesion Detection Rate ↑ | 63.18% | 59.12% |
| Mean Voxel Precision ↑ | 0.6304 | 0.6615 |
| Mean Voxel Recall ↑ | 0.6699 | 0.6602 |
| Mean FP Components/Case ↓ | 5.74 | 3.29 |

#### 선정 근거

두 실험의 Macro Dice 차이는 약 **0.0002**로 작았지만, `H-T01-E300`은 개별 병변 탐지율이 약 **4.1%p** 높았습니다. 대신 건당 오탐 병변이 약 **2.5개** 증가했습니다.

본 프로젝트에서는 뇌출혈 병변의 **미탐(False Negative)**이 후속 진료 판단에 미칠 수 있는 잠재적 위험을 중요하게 고려했습니다.

이에 따라 유사한 Dice 성능을 보이는 두 모델 중 병변을 더 많이 발견한 `H-T01-E300`을 최종 선택했습니다. 이는 프로젝트의 위험 우선순위에 근거한 모델 선정 결정이며, 임상 환경에서 더 안전함이 입증되었다는 뜻은 아닙니다.

선정된 모델의 최고 Validation Macro Dice는 **Epoch 268**에서 기록됐습니다(총 300 Epoch 학습).

향후 과제는 병변 탐지율을 유지하면서 FP를 줄이는 것입니다.

#### Evidence / Code

- [T01 vs T02 lesion-level selection notebook](./notebooks/hy/notebooks/25_T01_T02_lesion_level_final_selection.ipynb)
- [H-T01-E300 experiment configuration](./configs/experiments/hemorrhage/tuning/H-T01-E300.yaml)
- [Hemorrhage experiments](./configs/experiments/hemorrhage)

---

### 2.2 MRI — Ischemic Stroke Lesion Segmentation

**대표 앙상블 실험: `I-E01`**

DWI·ADC 영상을 대상으로 허혈성 병변 분할 모델을 개발했습니다.

`I-E01`에서는 nnU-Net 계열 `I-N120`과 SegResNet 계열 `I-S101`의 확률맵을 **Soft Voting(0.5:0.5)**으로 결합했습니다.

**Validation 38건**에서 병변 F2와 FP/scan 제약을 함께 고려하여 **Threshold 0.30**을 선택했습니다.

| Validation Metric | I-E01 |
| --- | ---: |
| Macro Dice ↑ | 0.8095 |
| Lesion Precision ↑ | 81.08% |
| Lesion Recall ↑ | 69.32% |
| Lesion F1 ↑ | 0.7474 |
| FP/scan ↓ | 1.11 |

이 실험은 분할 정확도뿐 아니라 병변 탐지와 오탐 간 균형을 고려한 앙상블 후보 선정 과정입니다.

Validation에서 정한 Threshold를 고정한 후 별도의 Test 데이터에서 평가해야 최종 일반화 성능을 보고할 수 있습니다.

#### Evidence / Code

- [I-E01 Soft-Voting experiment notebook](./notebooks/hy/Ischemic/06_I-E01_N120_S101_SoftVoting.ipynb)
- [Ischemia experiment configurations](./configs/experiments/ischemia)
- [Ensemble evaluation code](./scripts/ischemia/ensemble/evaluate_soft_voting.py)

---

### 2.3 MRA — Intracranial Aneurysm Segmentation

**대표 모델: `A-B04`**

MRA 영상에서 뇌동맥류 병변을 분할하기 위해 nnU-Net 기반 모델 및 Tversky-Focal 설정을 실험했습니다.

`A-B04`에서는 **Epoch 290**의 체크포인트가 최고 Full-volume Validation Macro Dice를 기록했습니다.

| Validation Metric | A-B04 |
| --- | ---: |
| Macro Dice ↑ | 0.5737 |
| Lesion Detection Rate ↑ | 82.50% (33/40) |

병변 크기별 분석에서는 **0.01 mL 미만의 병변 7개 중 1개**를 탐지했습니다.

전체 탐지율만으로 설명하기 어려운 극소 병변 탐지의 한계를 확인한 결과입니다.

#### Evidence / Code

- [A-B04 experiment notebook](./notebooks/hy/aneurysm_binary/04_A-B04_focal_tversky_baseline_sampling.ipynb)
- [A-B04 experiment configuration](./configs/experiments/aneurysm_binary/A-B04.yaml)
- [A-B04 evaluation code](./scripts/aneurysm_binary/validation/evaluate_a_b04_neighborhood.py)

---

## 3. My Contributions

3인 팀 프로젝트에서 **팀 리드와 AI 모델 개발·평가**를 중심으로 참여했습니다.

- **실험 설계 및 평가:** CT·MRI·MRA 모델별 실험 설정과 결과를 비교하고, Dice 외에 탐지율·오탐·병변 크기별 성능을 함께 분석했습니다.
- **의료영상 데이터 QA:** 영상·라벨 정합성, intensity 분포, NaN 및 이상값 등을 점검하여 학습 전 데이터 품질을 검증했습니다.
- **모델 선정 의사결정:** CT에서 성능 차이가 작은 후보들 사이의 미탐·오탐 상충관계를 분석하고 최종 모델 선정 근거를 정리했습니다.
- **실험 및 릴리스 관리:** MLflow 실험 이력, 버전별 모델 아티팩트, GCS 기반 릴리스 및 추론 서비스 연계 과정에 참여했습니다.
- **팀 협업:** 개발 일정·역할·모델 연계 이슈를 조율했습니다.

> 팀 프로젝트의 전체 성과와 개인 기여를 구분하여 기록했습니다. 기능별 상세 기여 범위는 관련 코드·실험 기록을 함께 참고해 주세요.

## 4. Development Workflow

1. **Data Preparation**  
   공개 의료영상 데이터셋 확인 및 학습용 구성

2. **Data Quality Assurance**  
   영상 메타데이터·라벨·전처리 일관성 검증

3. **Model Training**  
   nnU-Net, SegResNet 및 변형 모델 학습

4. **Validation & Selection**  
   분할 성능과 병변 탐지·오탐을 종합한 후보 선정

5. **Experiment Tracking**  
   실험 설정·체크포인트·성능 지표 이력 관리

6. **Model Release**  
   모델 아티팩트 버전 관리 및 추론 서비스 연계

데이터셋별 접근 조건과 라이선스를 준수하며, 원본 환자 영상 및 대용량 모델 가중치는 GitHub 저장소에 모두 포함하지 않습니다.

---

## 5. Repository Guide

| Path | Purpose |
| --- | --- |
| `src/brainon_ai` | 핵심 모델, 학습 엔진, 지표 및 추론 관련 코드 |
| `configs/experiments` | 모델별 실험 설정 파일 |
| `scripts` | 전처리·학습·평가·결과 분석 스크립트 |
| `notebooks` | 실험 실행 및 분석 기록 |
| `tests` | 기능 및 모델 검증 테스트 |
| `registry` | 데이터셋 및 모델 릴리스 메타데이터 |
| `docs` | 개발·평가·데이터 관리 기술 문서 |

---

## 6. Quick Start (Developers)

Python **3.11** 환경을 권장합니다. CUDA 환경에서는 현재 시스템과 호환되는 **PyTorch를 별도로 설치**한 뒤 프로젝트 패키지를 설치하세요.

```bash
git clone https://github.com/duddl6292/BrainOn-AI.git
cd BrainOn-AI
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,storage]"

# 외부 의료영상 데이터 없이 실행 가능한 점검
brainon data make-synthetic --task all
brainon smoke-test --task all
pytest -q
```

Windows PowerShell에서는 가상환경 생성·활성화를 다음과 같이 수행할 수 있습니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

실제 CT·MRI·MRA 학습 및 평가에는 별도 데이터셋 준비와 프로젝트 설정이 필요합니다. 전체 CLI 명령과 배포 관련 세부 사항은 [개발 문서](./docs)를 확인해 주세요.

## 7. Limitations & Future Work

- **False negatives vs. false positives**  
  병변 미탐을 줄이기 위한 설정은 오탐을 늘릴 수 있어, 임상 목적에 맞춘 운영 임계값과 후처리 전략이 필요합니다.

- **Small lesions**  
  극소 병변은 상대적으로 탐지가 어려웠으며, 병변 크기별 성능을 별도로 분석해야 합니다.

- **Generalization**  
  본 README의 수치는 해당 실험의 Validation 결과입니다. 외부 기관 데이터 및 독립 Test 코호트에서의 추가 검증 없이는 실제 임상 성능을 일반화할 수 없습니다.

- **Evaluation consistency**  
  병변 매칭(최소 겹침 기준·일대일 매칭 여부), 계산 단위(voxel/lesion), 지표 평균 방식 및 데이터 분할을 일관되게 관리할 필요가 있습니다.

---

**Acknowledgment:** 본 프로젝트는 팀원들의 공동 개발 결과이며, nnU-Net·MONAI 등 공개 연구 소프트웨어를 기반으로 수행한 실험을 포함합니다. 팀 원본은 [brainCDSS/model](https://github.com/brainCDSS/model)에서 확인할 수 있으며, 데이터셋과 사용한 원본 연구의 출처·라이선스는 각 데이터 및 구현 문서를 따릅니다.
