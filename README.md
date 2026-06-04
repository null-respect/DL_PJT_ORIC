## 실행 가이드 (clone → inference → evaluate)

이 문서는 이 레포에서 **ORIC-Bench에 대해 여러 VLM을 실행하고**, 모델별/전체 평가까지 수행하는 방법을 정리합니다.

### 원본 레포 / 라이선스

- **원본 프로젝트**: `ZhaoyangLi-1/ORIC` (CVPR 2026 ORIC)
- **라이선스**: MIT (`LICENSE` 참고)

---

### 0) git clone

```bash
git clone https://github.com/null-respect/DL_PJT_ORIC.git
cd ORIC
```

---

### 1) 환경 세팅

```bash
conda create -n oric python=3.10 -y
conda activate oric
bash setup.sh
```

---

### 2) 데이터셋(COCO 2014) 다운로드/압축해제

`infer.py`는 ORIC-Bench(`dataset/oric_bench.json`)의 `image` 필드(예: `COCO_val2014_000000420963.jpg`)를
`--image_dir` 아래에서 찾습니다. 아래 커맨드는 레포 기본 구조(`./dataset/val2014`, `./dataset/annotations`)를 맞춥니다.

```bash
cd dataset

# COCO 2014 validation images
wget -c http://images.cocodataset.org/zips/val2014.zip

# COCO 2014 train/val annotations
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# 압축 해제
unzip val2014.zip
unzip annotations_trainval2014.zip

cd ..
```

다운로드/해제 후 폴더 예시:

```text
dataset/
├── val2014/                       # COCO 2014 val 이미지
└── annotations/                   # COCO annotations
    ├── instances_val2014.json
    └── instances_train2014.json
```

---

### 3) VLM 추론 (단일 모델)

`infer.py`는 ORIC-Bench(`dataset/oric_bench.json`)를 읽어 **`predictions.json`**(README에서 요구하는 포맷)을 생성합니다.

```bash
python infer.py \
  --bench_path ./dataset/oric_bench.json \
  --image_dir ./dataset/val2014 \
  --model_family auto \
  --model_name_or_path "Qwen/Qwen3-VL-8B-Instruct" \
  --output_path ./predictions.json
```

#### 주요 옵션(`infer.py`, 단일)
- **`--bench_path`**: ORIC-Bench JSON 경로
- **`--image_dir`**: 이미지 폴더(예: `dataset/val2014`)
- **`--model_family`**: 어댑터 선택
  - `auto`: 모델 이름을 보고 자동 선택
  - `qwen3_vl`: Qwen3-VL 전용 어댑터
  - `hf_generic`: AutoModel/AutoProcessor 기반 범용 어댑터(많은 오픈 VLM에서 동작)
- **`--model_name_or_path`**: HF 모델 ID 또는 로컬 체크포인트 경로
- **`--output_path`**: 단일 모델 예측 저장 파일
- **`--num_prompts`**: ORIC 한 문항의 4개 프롬프트 중 몇 개를 사용할지(기본 1)
- **`--max_new_tokens`**: 생성 길이
- **`--temperature`**: 샘플링 온도(기본 0.0 = 결정적)
- **`--limit`**: 스모크 테스트용(처음 N개만 실행)

---

### 4) VLM 추론 (여러 모델 한 번에)

`--models_file`에 **모델 목록 JSON**을 주면 순차로 돌며 모델별 `predictions_<name>.json`을 저장합니다.

레포에 예시 목록이 포함되어 있습니다:
- `models_requested.json` (요청한 12개 모델)
- `models_paper.json` (논문 Table 12 전체: LVLM 18 + detector 2)

```bash
python infer.py \
  --bench_path ./dataset/oric_bench.json \
  --image_dir ./dataset/val2014 \
  --models_file ./models_requested.json \
  --output_dir ./preds_requested \
  --resume
```

#### 주요 옵션(`infer.py`, 멀티)
- **`--models_file`**: JSON 목록(예: `models_requested.json`)
- **`--output_dir`**: 모델별 예측 파일 저장 폴더
- **`--resume`**: `predictions_<model>.json`이 이미 있으면 그 모델은 스킵
- **`--hf_token`**: HF gated 모델 접근이 필요할 때 토큰을 넘김  
  - 내부적으로 `HUGGINGFACE_HUB_TOKEN`을 설정합니다.
- **`--load_in_4bit`**: 16GB GPU 등 VRAM이 부족할 때 4-bit 양자화로 로드 (bitsandbytes)
- **`--dtype`**: `bfloat16` / `float16` 등 (4-bit 미사용 시)

모델 JSON 항목별로 `load_in_4bit`, `dtype`, `run: false`, `note`를 줄 수 있습니다.

> 참고: HF에서 접근 제한(gated)인 모델(예: 일부 Llama 계열)은 **모델 페이지에서 승인**이 필요합니다. 

> 16GB GPU에서 연속 실행 시 이전 모델 VRAM이 남으면 OOM이 날 수 있습니다. `infer.py`는 모델마다 GPU 메모리를 해제합니다. 큰 모델은 `--load_in_4bit` 또는 JSON의 `"load_in_4bit": true`를 권장합니다.

---

### 5) (선택) “모델별로 추론 직후 바로 평가”까지 같이 수행

멀티 모델 실행 시 `--evaluate_each`를 켜면, 각 모델 추론이 끝날 때마다 평가 결과를 바로 저장합니다.

```bash
python infer.py \
  --models_file ./models_requested.json \
  --output_dir ./preds_requested \
  --resume \
  --evaluate_each
```

결과:
- 예측: `preds_requested/predictions_<model>.json`
- 평가: `preds_requested/eval/<model>/results.json`

옵션:
- **`--eval_dir`**: 평가 저장 루트 폴더(기본은 `<output_dir>/eval`)

---

### 7) 평가 (단일 예측 파일)

```bash
python evaluate.py \
  --result_path ./predictions.json \
  --output_folder ./results_single
```

---

### 8) 평가 (여러 모델 예측 파일을 한 번에)

`--results_dir` 아래에서 `predictions_*.json`을 전부 찾아 모델별 폴더로 결과를 저장하고, `summary.json`도 생성합니다.

```bash
python evaluate.py \
  --results_dir ./preds_requested \
  --output_folder ./eval_requested
```

#### (권장) 모델 목록(`models_file`) 기준으로 “정확히 매핑”해서 평가

`infer.py`가 생성한 파일명이 달라도, 모델 목록의 `name` 기준으로 `predictions_<name>.json`을 찾아서 결과를 저장합니다.

```bash
python evaluate.py \
  --results_dir ./preds_requested \
  --models_file ./models_requested.json \
  --output_folder ./eval_requested
```

옵션:
- **`--strict`**: 목록에 있는 모델의 예측 파일이 누락되면 즉시 실패(누락 감지용)
- **`--pattern`**: glob 패턴(기본 `predictions_*.json`)

---

### 출력 포맷(예측 JSON)

`infer.py`가 생성하는 포맷은 아래와 같습니다.

```json
[
  {"question_id": "1", "predicted_answer": "yes", "solution": "yes"},
  {"question_id": "2", "predicted_answer": "no",  "solution": "no"}
]
```

