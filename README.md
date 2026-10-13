# MI-VA: MI-CDM AI-ECG 검증 도구

- **model.json** (`registry/*/*.json`, 모델 카드): 모델의 입력 규격(lead, 샘플링, 길이, 단위),
  출력 형식, 가중치 위치와 sha256, 실행 env.
- **study.yaml** (`studies/*/study.yaml`): 어떤 환자의 어떤 ECG를, 어떤 정답과 지표로 평가할지.

두 파일을 기관에 맞게 두면 나머지는 5단계로 실행된다.

```
MI-CDM ─▶ retrieve ─▶ profile ─▶ preprocess ─▶ models ─▶ evaluate
          ECG·라벨     사람 단위    카드 규격으로    모델별 예측    test 지표,
          추출        dev/test 분할  파형 변환      (env별 실행)   95% CI, 그림
```

## 1. 검증 대상

| task | study | 라벨 | 모델 (card `model_id`) | 방식 |
|---|---|---|---|---|
| LVEF | `lvef` | ECG 날짜 ±7일 안 가장 가까운 LVEF (measurement 3027172, %). 분류 LVEF ≤ 40, 회귀 LVEF % | `heartwise-lvef-binary`, `heartwise-lvef-regression` | 공개 가중치 그대로 |
| | | | `ecgfounder`, `xecg` | backbone 고정, dev에서 linear probe 학습 |
| ECG 나이 | `ecg-age` | ECG 연도 − 출생연도 (person.year_of_birth), 사람당 ECG 1건 | `lima-ecg-age`, `singstad-ecg-age` | 공개 가중치 그대로 |
| 1년 사망 | `mortality` | ECG 후 365일 안 사망 (death), 추적 종료 = min(ECG + 365일, 마지막 방문 종료 + 365일) | `cavalab-deepsurv-code15`, `cavalab-mtlr-code15` (나이·성별 사용) | 공개 가중치 그대로 |
| 혈청 칼륨 | `potassium` | 같은 날 혈청 칼륨 (measurement 3023103, mmol/L). 회귀, K < 3.5 / K > 5.5 판별 | `kardionet-k-12lead`, `vonbachmann-k` | 공개 가중치 그대로 |

모든 study는 사람 단위로 dev 80% / test 20%를 나누고, 지표는 test에서 사람 단위 bootstrap
2,000회로 95% CI를 낸다.

### MIMIC-IV-ECG 참조 결과 (test, 95% CI)

기관 결과와 비교하기 위한 값이다.

| task | 모델 | 결과 |
|---|---|---|
| LVEF | HeartWise | AUROC 0.844 (0.834–0.854), MAE 8.33, R² 0.41 |
| | ECGFounder (linear probe) | AUROC 0.867 (0.858–0.875), MAE 8.96, R² 0.37 |
| | xECG (linear probe) | AUROC 0.890 (0.881–0.899), MAE 8.41, R² 0.44 |
| ECG 나이 | Lima | MAE 9.90 (9.82–9.98), R² 0.58 |
| | Singstad | MAE 10.04 (9.86–10.25), R² −0.09 |
| 1년 사망 | cavalab DeepSurv | C-index 0.785 (0.774–0.796), AUROC@1년 0.788 |
| | cavalab MTLR | C-index 0.811 (0.799–0.821), AUROC@1년 0.814 |
| 혈청 칼륨 | von Bachmann | MAE 0.357, AUROC K<3.5 0.760, K>5.5 0.851 |
| | Kardio-Net | MAE 0.429, AUROC K<3.5 0.713, K>5.5 0.787 |

## 2. 필요한 환경

| 항목 | 조건 |
|---|---|
| OS | Linux x86_64 (Ubuntu 등). 번들의 env는 Linux 전용이다. 코드 자체는 OS와 무관하다. |
| conda | env 압축을 풀고 `conda-unpack`을 실행하는 데만 쓴다 (Miniconda/Anaconda 무관). |
| GPU | 선택. NVIDIA GPU와 드라이버가 있으면 torch 모델이 GPU로 돈다. 없으면 CPU로 돌고, **xECG만 제외된다** (아래 6절). |
| DB | MI-CDM이 적재된 **PostgreSQL** 또는 **SQL Server (MSSQL)**. `image_occurrence`, `person`, `measurement`, `death`, `visit_occurrence`를 읽기만 한다. 드라이버(psycopg, pymssql)는 env에 들어 있어 ODBC 드라이버 설치가 필요 없다. |
| DICOM | `image_occurrence.local_path`가 가리키는 12-lead ECG DICOM 파일이 서버에서 읽을 수 있는 위치에 있어야 한다. |
| 인터넷 | 필요 없다. 실행에 필요한 env, 가중치, 모델 코드가 모두 번들에 들어 있다. |
| 디스크 | 번들 (크기는 번들 제작 후 기입) + 실행 결과. 변환된 파형 텐서가 큰 study는 수십~100 GB를 쓴다 (`MIVAL_RUNS`로 위치 지정). |

## 3. 설치 (내부망 반입 번들)

반입 번들 `miva-bundle-261013.tar.gz`를 풀면 아래 구조가 된다. 이 폴더가 `MIVAL_HOME`이다.

```
$MIVAL_HOME/
├── miva/         이 저장소
├── envs/         conda-pack env 압축 3개
│   ├── mival-torch.tar.gz     torch (CUDA 빌드, GPU 없으면 CPU로 동작). retrieve·profile·preprocess·evaluate와 torch 모델 7개
│   ├── mival-keras27.tar.gz   TensorFlow/Keras 2.7 (CPU). singstad-ecg-age
│   └── mival-xecg.tar.gz      xECG 전용 (GPU 필요, 커널 미리 컴파일됨)
├── models/       가중치 (카드의 weights.uri 위치 그대로)
├── code/         저자 공개 코드 (카드의 code_path 위치 그대로)
├── cache/        xECG 컴파일된 커널
├── config/       micdm.env, site.yaml (기관에서 만든다, 4절)
└── SHA256SUMS
```

```bash
# 1) 압축 해제 (경로는 기관 사정에 맞게)
mkdir -p /opt/mival && tar -xzf miva-bundle-261013.tar.gz -C /opt/mival
export MIVAL_HOME=/opt/mival            # 셸 설정(~/.bashrc 등)에 넣어 둔다
cd $MIVAL_HOME && sha256sum -c SHA256SUMS

# 2) env 3개 풀기 (CXR 나이예측 모델과 같은 conda-pack 방식)
for e in mival-torch mival-keras27 mival-xecg; do
  mkdir -p envs/$e && tar -xzf envs/$e.tar.gz -C envs/$e
  envs/$e/bin/conda-unpack
done
```

GPU가 없는 서버는 `mival-xecg`를 풀지 않아도 된다.

## 4. 기관 설정

기관이 만드는 파일은 `$MIVAL_HOME/config/` 아래 2개뿐이다. 저장소 안의 study.yaml과 카드는 고치지 않는다.

| 파일 | 내용 | 양식 |
|---|---|---|
| `config/micdm.env` | DB 접속 정보 (비밀번호 포함) | `miva/config/micdm.env.example` |
| `config/site.yaml` | 기관 식별자, DBMS, schema, ECG concept, DICOM 경로, 코호트 이름 | `miva/config/site.yaml.example` |

study 4개는 모두 첫 줄의 `site_config: ${MIVAL_HOME}/config/site.yaml`로 같은 기관 설정을 읽는다.

### 4.1 DB 접속 파일 (`micdm.env`)

```bash
mkdir -p $MIVAL_HOME/config
cp $MIVAL_HOME/miva/config/micdm.env.example $MIVAL_HOME/config/micdm.env
chmod 600 $MIVAL_HOME/config/micdm.env      # DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD 채우기
```

PostgreSQL과 SQL Server 모두 같은 키를 쓴다. 어느 DB인지는 `site.yaml`의 `retrieve.dbms`로 정한다.
PostgreSQL은 libpq 이름(`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`)도 읽는다. 값 뒤에 주석을 붙이지 않는다.

접속 값은 로그와 결과 manifest에 남지 않는다. 파일 경로만 기록된다. 읽기 권한만 있는 계정이면 된다.

### 4.2 기관 설정 (`site.yaml`)

```bash
cp $MIVAL_HOME/miva/config/site.yaml.example $MIVAL_HOME/config/site.yaml
```

```yaml
site: my-site
retrieve:
  dbms: postgresql
  dsn_env: ${MIVAL_HOME}/config/micdm.env
  schema: cdm
  modality_concept_id: 4145308
  local_path_root: null
models:
  cohort_sources: [my-site-ecg]
```

| 항목 | 값 |
|---|---|
| `site` | 기관 식별자 (예: `severance`). 결과 manifest에 기록된다. |
| `retrieve.dbms` | `postgresql` 또는 `sqlserver`. |
| `retrieve.dsn_env` | DB 접속 파일. 기본 `${MIVAL_HOME}/config/micdm.env`. |
| `retrieve.schema` | MI-CDM 테이블이 있는 schema. SQL Server는 다른 database의 schema를 `CDM2025.dbo`처럼 쓸 수 있다. |
| `retrieve.modality_concept_id` | `image_occurrence`에서 12-lead ECG를 가리키는 concept. 기본 4145308. |
| `retrieve.local_path_root` | 기본 `null`. 아래 설명 참고. |
| `models.cohort_sources` | 이 코호트의 데이터셋 이름 (예: `severance-ecg`). 모델 사전학습 데이터와 겹치는지 표시하는 데 쓴다. |

`site.yaml`에 있는 항목을 study.yaml에도 적으면 어느 값을 쓸지 모호하므로 실행이 멈춘다.

**DICOM 경로 규칙.**

- `local_path_root: null` (기본): `image_occurrence.local_path`를 **그대로** 읽는다.
  CDM의 local_path가 이 서버에서 읽을 수 있는 경로이면 이것만 확인하면 된다.
- `local_path_root: /경로`: CDM을 만든 곳과 파일 위치가 다를 때. local_path에서 `/files/` 뒤 부분을 잘라
  `local_path_root/files/` 아래에 다시 붙인다 (`/files/`가 없는 경로는 제외).

```
local_path      = /원래/위치/files/p1000/p10000032/s1/1.dcm
local_path_root = /mnt/ecg
읽는 파일       = /mnt/ecg/files/p1000/p10000032/s1/1.dcm
```

local_path가 비어 있거나 파일이 없는 ECG는 제외되고 제외 사유가 기록된다.
`sql/check_cohort.<dbms>.sql`의 1번 쿼리에서 `example_path`가 이 서버에서 열리는지 먼저 확인한다.

### 4.3 study.yaml (바꾸지 않음)

`studies/*/study.yaml`에는 task 정의(라벨 concept, window, 분할 비율, 모델, 지표)만 있다. 참조 결과와 같은 설정으로
비교하기 위해 **바꾸지 않는다**. 각 항목 설명은 [`docs/study-yaml-reference.md`](docs/study-yaml-reference.md)에 있다.

## 5. 실행 전 확인

```bash
cd $MIVAL_HOME/miva
PY=$MIVAL_HOME/envs/mival-torch/bin/python

# 1) 기관 설정과 반입물 확인: site.yaml·micdm.env, 카드가 가리키는 가중치(sha256)·코드·env
$PY scripts/check_bundle.py            # 마지막 줄 "all ok"

# 2) 코호트 규모 확인: sql/check_cohort.postgresql.sql 또는 check_cohort.sqlserver.sql 을
#    DB 클라이언트에서 실행 (schema, concept 바꿔서)
```

## 6. 실행

```bash
cd $MIVAL_HOME/miva
scripts/run_study.sh lvef
scripts/run_study.sh ecg-age
scripts/run_study.sh mortality
scripts/run_study.sh potassium
```

- study 하나가 5단계를 차례로 돈다. models 단계는 카드의 `runtime.env`별로 나눠 돌고
  (`mival-torch`, `mival-keras27`, `mival-xecg`), 예측을 모아 evaluate한다.
- **GPU가 없으면** torch 모델은 CPU로 돌고, LVEF의 xECG arm 2개는 건너뛴다(로그에 `skip xecg`).
  나머지 결과는 그대로 나온다.
- 끝난 단계는 다시 돌지 않는다. 중간에 실패하면 원인을 고친 뒤 같은 명령을 다시 실행하면
  실패한 단계부터 이어서 돈다.
- 결과 위치는 `MIVAL_RUNS` (기본 `$MIVAL_HOME/runs`). 디스크가 넉넉한 곳으로 지정할 수 있다.
- 오래 걸리므로 `tmux`나 `nohup` 안에서 실행한다.

### 소요 시간 (MIMIC-IV-ECG, GPU 1장, 참고용)

| study | 코호트 | 시간 |
|---|---|---|
| lvef | 1인 다건, test 36,170건 | DICOM 읽기 43분, preprocess 33분, models: torch env 약 4시간 50분, xECG 약 3시간 40분, evaluate 24분 |
| ecg-age | 사람당 1건, 158,566건 | 약 40분 |
| mortality | 사람당 1건, 156,484건 | 기록 없음 (ecg-age와 비슷한 규모) |
| potassium | 사람당 1건, 124,167건 | 기록 없음 |

CPU 서버 시간은 측정하지 않았다. models 단계가 GPU보다 훨씬 오래 걸린다.

## 7. 결과 확인

`runs/<study>/_run/run.log` 마지막 줄에 evaluate 결과 폴더가 찍힌다.

```
runs/<study>/
├── retrieve/<hash>/   artifacts/: cohort_index.parquet, retrieve_summary.json, 실행한 SQL
├── profile/<hash>/    cohort_split.parquet (dev/test), profile_summary.json
├── preprocess/<hash>/ 모델 규격으로 변환된 파형
├── models/<hash>/     predictions/*.parquet (ECG별 예측)
├── evaluate/<hash>/   metrics_long.parquet, comparisons.parquet, figures/*.png
├── exclusions/        제외된 ECG와 사유 (단계별)
└── _run/              단계별 로그
```

각 단계 폴더(`<stage>/<hash>/`)에는 `artifacts/`, `logs/`, `manifest.json`(입력·설정·코드·env 기록)이 있고,
제외된 ECG와 사유는 study 단위 제외 기록(exclusion ledger)에 쌓인다.

```bash
$MIVAL_HOME/envs/mival-torch/bin/python - <<'PY'
import glob, os, pandas as pd
runs = os.environ.get("MIVAL_RUNS", os.environ["MIVAL_HOME"] + "/runs")
t = pd.read_parquet(max(glob.glob(runs + "/lvef/evaluate/*/artifacts/metrics_long.parquet"), key=os.path.getmtime))
t = t[(t.split == "test") & (t.subgroup == "all")]
print(t[["model_id", "label_def", "metric", "value", "ci_lo", "ci_hi", "n"]].to_string(index=False))
PY
```

**기관 밖으로 낼 수 있는 것은 집계 결과뿐이다**: `evaluate/`의 지표표와 그림, 각 단계의 `manifest.json`과
summary json. `cohort_index`, `cohort_split`, 변환된 파형, `predictions`는 환자·ECG 단위 자료이므로 내보내지
않는다.

## 8. 전체 파일 목록

### 저장소 (`$MIVAL_HOME/miva/`)

```
README.md                         이 문서
pyproject.toml                    패키지 정의 (mival), 의존성
.gitignore
config/
  micdm.env.example               DB 접속 파일 양식 (4.1절)
  site.yaml.example               기관 설정 양식: site, DBMS, schema, DICOM 경로 (4.2절)
sql/
  check_cohort.postgresql.sql     실행 전 코호트 규모 확인 쿼리, PostgreSQL (파이프라인은 쓰지 않음)
  check_cohort.sqlserver.sql      같은 쿼리, SQL Server
scripts/
  run_study.sh                    study 하나를 5단계로 실행 (6절)
  check_bundle.py                 기관 설정과 카드의 가중치·sha256·코드·env 확인 (5절)
docs/
  study-yaml-reference.md         study.yaml 항목별 설명

studies/                          ── study.yaml: 무엇을 어떻게 평가할지
  lvef/study.yaml                 LVEF 분류(≤40)·회귀, 모델 4개 6 arm
  ecg-age/study.yaml              ECG 나이 회귀, 모델 2개
  mortality/study.yaml            1년 사망 생존분석, 모델 2개
  potassium/study.yaml            혈청 칼륨 회귀·판별, 모델 2개

registry/                         ── model.json: 모델 카드
  models/                         LVEF study의 카드
    heartwise-lvef-binary.json      HeartWise EfficientNetV2, LVEF ≤ 40 분류 (TorchScript)
    heartwise-lvef-regression.json  HeartWise EfficientNetV2, LVEF % 회귀
    ecgfounder.json                 ECGFounder 12-lead backbone (linear probe)
    xecg.json                       xECG xLSTM backbone (linear probe, GPU 필요)
  age/
    lima-ecg-age.json               Lima CODE ResNet, 나이 회귀
    singstad-ecg-age.json           Singstad InceptionTime, 나이 회귀 (Keras 2.7)
  survival/
    cavalab-deepsurv-code15.json    cavalab ResNet DeepSurv (CODE-15)
    cavalab-mtlr-code15.json        cavalab ResNet MTLR, 나이·성별 공변량 (CODE-15)
  potassium/
    kardionet-k-12lead.json         Kardio-Net 12-lead 칼륨 회귀
    vonbachmann-k.json              von Bachmann 칼륨 회귀, 5-fold 앙상블
  wrappers/                       카드 builder가 쓰는 모델별 전처리·출력 모듈
    README.md
    mival_wrap_cavalab.py           cavalab 2종: 저장된 lead별 정규화, 4096 padding
    mival_wrap_potassium.py         Kardio-Net 기저선 제거·wavelet·정규화, von Bachmann high-pass
    mival_wrap_singstad.py          Singstad 구조 재생성, int32 절사

src/mival/                        ── 엔진 (모델·기관 이름이 코드에 없다)
  __init__.py
  modelcard.py                    카드 읽기, ${MIVAL_HOME} 치환
  placeholders.py                 ${NAME} → 환경변수 값 (카드, site.yaml 공용)
  contract.py                     입력 규격과 제외 사유 어휘
  compiler.py                     입력 규격 → 전처리 recipe
  ops.py                          전처리 연산 (lead 선택, resample, crop/pad, 단위, scaling)
  signal.py                       ECG 메모리 표현
  perturbation.py                 강건성 실험 격자 (resample, duration, lead dropout, 진폭)
  gates.py                        누수·오염 검사 (사전학습 데이터 겹침)
  threshold.py                    분류 임계값 정책 (dev에서 정하고 test에 적용)
  figures.py                      ROC/PR, calibration, decision curve, scatter 등
  synthetic.py                    테스트용 합성 ECG
  adapters/                       모델 실행 backend
    __init__.py                     backend 선택
    base.py                         adapter 인터페이스
    torch_adapter.py                PyTorch (cuda 없으면 cpu)
    keras_adapter.py                Keras 2.7
    _training.py                    linear probe 학습 (backend 공통)
    _attribution.py                 attribution (backend 공통)
  decode/                         모델 출력 → ECG별 값
    __init__.py
    outputs.py                      회귀·생존·확률 외 출력 해석
    transforms.py                   출력 → 라벨 척도 변환
    frames.py                       짧은 창 모델을 긴 기록에 적용
    intervals.py                    delineation mask → PR/QRS/QT
  metrics/                        지표
    __init__.py                     지표 정의 목록
    _common.py
    discrimination.py               AUROC, AUPRC, 민감도·특이도 등
    calibration.py                  calibration
    utility.py                      net benefit
    regression.py                   MAE, R², bias, cut별 AUROC
    survival.py                     C-index, 시점 AUROC
    bootstrap.py                    사람 단위 bootstrap CI, 모델 비교
  pipeline/                       실행 틀
    __init__.py
    cli.py                          `mival run <stage>` 명령
    study.py                        study.yaml 읽기, site.yaml 합치기
    stage.py                        단계 등록과 실행 (끝난 단계 건너뛰기)
    layout.py                       결과 폴더 구조
    manifest.py                     실행 기록 (입력, 설정, 코드, env)
    ledger.py                       제외 기록
    hashing.py                      sha256, config_hash
    runkey.py                       결과 좌표 (model, label, split, subgroup …)
    tables.py                       parquet 입출력
  stages/                         5단계 구현
    __init__.py
    retrieve.py                     MI-CDM(PostgreSQL/SQL Server)에서 ECG·라벨 추출, DICOM 경로 연결
    profile.py                      사람 단위 dev/test 분할, 양성 수 확인
    preprocess.py                   카드별 파형 변환
    models.py                       arm별 예측, linear probe 학습
    evaluate.py                     test 지표, CI, 그림
    _predictions.py                 예측 읽기 (evaluate 공용)
    misclassify.py                  오분류 사례 분석 (파이프라인에서 제외됨, 실행되지 않음)
    sql/
      retrieve_ecg_label.sql          ECG + 가장 가까운 measurement 라벨 (PostgreSQL)
      retrieve_ecg_label.sqlserver.sql  같은 쿼리, SQL Server (outer apply, datediff)
      retrieve_ecg_person.sql         ECG + 출생연도·사망·마지막 방문 (나이·사망 라벨, 두 DBMS 공용)
      label_snapshot.sql              라벨 원천 스냅샷

```

### 반입 번들에만 있는 것 (GitHub에는 없음)

```
envs/mival-torch.tar.gz           conda-pack env: Python 3.10, torch (CUDA), pydicom, psycopg, pymssql 등
envs/mival-keras27.tar.gz         conda-pack env: Python 3.9, TensorFlow/Keras 2.7 (CPU)
envs/mival-xecg.tar.gz            conda-pack env: Python 3.12, torch 2.8 (CUDA), xlstm
models/                           가중치 (카드의 weights.uri), 아래 목록
  heartwise-lvef/                   lvef_under_equal_40.pt, LVEF_MSE_SL.pt
  ecgfounder/                       12_lead_ECGFounder.pth
  xecg/                             model.safetensors, config.json, xECG.py
  lima-ecg-age/model/               model.pth
  singstad-ecg-age/                 model_weights_inception_both genders.h5
  cavalab-deepsurv-code15/          mival_state_dict.pt
  cavalab-mtlr-code15/              mival_state_dict.pt
  kardionet-k-12lead/               mival_state_dict.pt
  vonbachmann-k/regression_models/potassium/model_{0..4}/model.pth
code/                             저자 공개 코드 (카드의 code_path, wrapper가 읽는 위치)
  ecgfounder/  deepecg-docker/  ecg-age-prediction/  ECG-age/
  ecg-survival-benchmark/  hyperkalemia/  ecg-electrolyte-regression/
cache/torch_extensions/           xECG 컴파일된 CUDA 커널
SHA256SUMS
```

## 9. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `${MIVAL_HOME} is used but MIVAL_HOME is not set` | `export MIVAL_HOME=...` (셸 설정에 넣기) |
| `site_config ... does not exist` | `$MIVAL_HOME/config/site.yaml` 만들기 (4.2절) |
| `... set both here and in .../site.yaml` | study.yaml에 기관 항목을 적었다. study.yaml은 원래대로 두고 site.yaml에만 적는다 |
| `check_bundle.py`에서 missing / sha256 mismatch | 번들 압축 해제 위치, `sha256sum -c SHA256SUMS` |
| retrieve에서 ECG가 0건 | site.yaml의 `dbms`, `schema`, `modality_concept_id`, `local_path_root`와 local_path가 이 서버에서 열리는지 (4.2절) |
| DB 접속 실패 | `micdm.env`의 키 이름과 값 (값 뒤 주석 금지), 방화벽, SQL Server는 TCP/IP 활성화와 포트 1433 |
| `min_test_positives` 오류로 profile 실패 | test 양성 수가 100건 미만. 코호트 규모 문제이므로 연구실에 알린다 |
| models [mival-xecg] 실패 | GPU, 드라이버. GPU가 없으면 정상적으로 건너뛴다 |

문의: 연세대학교 의과대학 의생명시스템정보학교실
