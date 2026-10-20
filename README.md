# MI-VA: MI-CDM AI-ECG 검증 도구

- **model.json** (`registry/*/*.json`, 모델 카드): 모델의 입력 규격(lead, 샘플링, 길이, 단위),
  출력 형식, 가중치 위치와 sha256, 실행 env.
- **study.yaml** (`studies/*/study.yaml`): 어떤 환자의 어떤 ECG를, 어떤 정답과 지표로 평가할지.

두 파일을 기관에 맞게 두면 나머지는 6단계로 실행된다.

```
MI-CDM ─▶ retrieve ─▶ profile ─▶ preprocess ─▶ describe ─▶ models ─▶ evaluate
          ECG·라벨     사람 단위    카드 규격으로   Table 1,     모델별 예측    test 지표,
          추출        dev/test 분할  파형 변환     제외·STARD    (env별 실행)   95% CI, 그림
```

기관은 마지막에 집계 결과만 zip 하나로 묶어(`export_results`) 연구실에 보낸다 (7절).

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

아래 값은 이 공유본과 같은 코드·study.yaml로 MIMIC-IV-ECG에서 돌린 참조 실행(2026-10-20) 값이다.
저칼륨 기준은 K ≤ 3.5에서 **K < 3.5**로 바뀌었다. LVEF study는 LVEF 값으로 제외하지 않는다(`implausible_below: null`).

| task | 모델 | 결과 |
|---|---|---|
| LVEF | HeartWise | AUROC 0.845 (0.835–0.855), MAE 8.39 (8.22–8.57), R² 0.42 |
| | ECGFounder (linear probe) | AUROC 0.867 (0.858–0.876), MAE 9.06 (8.89–9.23), R² 0.37 |
| | xECG (linear probe) | AUROC 0.892 (0.884–0.900), MAE 8.42 (8.27–8.58), R² 0.45 |
| ECG 나이 | Lima | MAE 9.90 (9.82–9.98), R² 0.58 |
| | Singstad | MAE 10.04 (9.86–10.25), R² −0.09 |
| 1년 사망 | cavalab DeepSurv | C-index 0.785 (0.774–0.796), AUROC@1년 0.788 |
| | cavalab MTLR | C-index 0.811 (0.799–0.821), AUROC@1년 0.814 |
| 혈청 칼륨 | von Bachmann | MAE 0.357, AUROC K<3.5 0.768 (0.756–0.779), K>5.5 0.851 |
| | Kardio-Net | MAE 0.429, AUROC K<3.5 0.719 (0.708–0.732), K>5.5 0.787 |

## 2. 필요한 환경

| 항목 | 조건 |
|---|---|
| OS | Linux x86_64 (Ubuntu 22.04 등) 또는 Windows Server 2022 (Docker 없이 직접 실행). 실행 진입점은 두 OS에서 같은 Python 코드를 부른다 (`.sh`와 `.cmd`는 인터프리터만 찾는다). 차이는 아래 6절. |
| conda | env 압축을 풀고 `conda-unpack`을 실행하는 데만 쓴다 (Miniconda/Anaconda 무관). |
| GPU | 선택. NVIDIA GPU와 드라이버가 있으면 torch 모델이 GPU로 돈다. 없으면 CPU로 돌고, **xECG만 제외된다** (아래 6절). 여러 장이면 `site.yaml`의 `gpus`에 번호를 적는다 (4.2절). |
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
gpus: []                  # 예: 2장이면 [0, 1], 8장이면 [0, 1, 2, 3, 4, 5, 6, 7]
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
| `gpus` | models 단계에서 쓸 GPU 번호. arm 하나가 GPU 하나를 쓰고 동시에 `len(gpus)`개가 돈다. 비우면 GPU가 있을 때 `[0]`, 없으면 CPU에서 하나씩. 실행 방식일 뿐이라 결과는 GPU 수와 상관없이 같다. |
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
$PY scripts/check_bundle.py            # 마지막 줄 "all ok" (Windows는 아래 안내)

#    Windows: %MIVAL_HOME%\envs\mival-torch\python.exe scripts\check_bundle.py

# 2) 코호트 규모 확인: sql/check_cohort.postgresql.sql 또는 check_cohort.sqlserver.sql 을
#    DB 클라이언트에서 실행 (schema, concept 바꿔서)
```

`check_bundle.py`는 아래도 문제로 센다: `site`와 `cohort_sources`가 양식 값(`my-site`) 그대로인 경우,
`gpus`에 적은 번호를 torch가 못 보는 경우, torch가 드라이버보다 새 CUDA로 빌드된 경우, 가중치 sha256 불일치.

**Windows에서 나오는 줄 (보라매).** LVEF의 `xecg` 카드는 Linux에서 CUDA 커널을 컴파일해 쓰지만, Windows에서는
카드의 `runtime.windows` 설정에 따라 컴파일이 필요 없는 PyTorch 구현(sLSTM `vanilla` backend)으로 돈다.
가중치는 실행할 때 그 구현의 배열 순서로 바꿔 읽는다. check_bundle은 이것을 문제가 아니라 안내 줄로 보여 준다.

```
runtime.windows override applies on win32: ...
```

MIMIC 참조 서버에서 두 구현을 비교했을 때 feature cosine 유사도는 최소 0.99985였고, LVEF xECG probe의
AUROC·MAE는 소수 넷째 자리까지 같았다. 대신 xECG backbone이 약 12배 느려 xECG arm이 더 오래 걸린다.
어느 구현으로 돌았는지는 반출 묶음의 `export_manifest.json` `arms`에 기록된다.

**Windows 긴 경로.** 결과 파일 중 가장 긴 것은 `<MIVAL_RUNS>\<study>\models\<hash>\artifacts\predictions\<run_key>.parquet`이다.
check_bundle이 이 길이를 어림해 260자를 넘으면 문제로 센다 (`LongPathsEnabled`가 꺼져 있을 때만).
둘 중 하나로 푼다: 레지스트리 `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`의 `LongPathsEnabled`를 1로 켜거나
(재부팅), `C:\mival`처럼 짧은 경로에 설치한다. models 단계가 몇 시간 뒤에 경로 때문에 실패하는 것을 막으려는 확인이다.

## 6. 실행

Linux:

```bash
cd $MIVAL_HOME/miva
scripts/run_study.sh lvef
scripts/run_study.sh ecg-age
scripts/run_study.sh mortality
scripts/run_study.sh potassium
scripts/export_results.sh            # 4개가 끝난 뒤. 회신할 zip을 만든다 (7절)
```

Windows (cmd):

```bat
cd %MIVAL_HOME%\miva
scripts\run_study.cmd lvef
scripts\run_study.cmd ecg-age
scripts\run_study.cmd mortality
scripts\run_study.cmd potassium
scripts\export_results.cmd
```

`.sh`와 `.cmd`는 mival-torch 인터프리터를 찾아 `run_study.py`·`export_results.py`를 부를 뿐이라 두 OS가 같은 단계를 돈다.

- study 하나가 6단계(retrieve, profile, preprocess, describe, models, evaluate)를 차례로 돈다. models 단계는
  카드의 `runtime.env`별로 나눠 돌고 (`mival-torch`, `mival-keras27`, `mival-xecg`), 예측을 모아 evaluate한다.
  `site.yaml`의 `gpus`가 여러 장이면 arm을 GPU 수만큼 동시에 돌린다.
- **GPU가 없으면** torch 모델은 CPU로 돌고, LVEF의 xECG arm 2개는 건너뛴다(로그에 `skip xecg`).
  나머지 결과는 그대로 나온다.
- 끝난 단계는 다시 돌지 않는다. 중간에 실패하면 원인을 고친 뒤 같은 명령을 다시 실행하면
  실패한 단계부터 이어서 돈다.
- 결과 위치는 `MIVAL_RUNS` (기본 `$MIVAL_HOME/runs`). 디스크가 넉넉한 곳으로 지정할 수 있다.
- 오래 걸리므로 Linux는 `tmux`나 `nohup`, Windows는 끊겨도 계속 도는 세션(예: 작업 스케줄러, 또는 RDP 세션 유지) 안에서 실행한다.
- 기관 안의 지표는 **전처리 recipe별로** 계산된다. 모델마다 입력 규격이 달라 recipe가 다르고,
  지표표의 `recipe_id` 열로 구분한다. 서로 다른 recipe의 값을 한 모델의 값으로 합치지 않는다.

### linear probe feature 캐시

xECG·ECGFounder linear probe는 같은 dev ECG를 학습률마다 반복해서 backbone에 통과시킨다. 이 backbone 출력
(feature)을 메모리에 담아 두어 반복을 줄인다. 켜져 있는 것이 기본이며 결과 값은 바뀌지 않는다.

| 환경변수 | 기본 | 뜻 |
|---|---|---|
| `MIVAL_FEATURE_CACHE` | 켜짐 | `0`, `false`, `off`, `no`이면 끈다 |
| `MIVAL_FEATURE_CACHE_MAX_GB` | 4 | arm 하나당 캐시 상한 (GiB). 넘으면 새 행은 계산만 하고 저장하지 않는다 |

arm이 동시에 돌 때는 arm마다 상한이 따로 적용된다. `gpus`가 8장이면 최대 32 GiB RAM을 쓸 수 있으므로
(4 × 8), 서버 메모리가 모자라면 `MIVAL_FEATURE_CACHE_MAX_GB`를 줄이거나 `MIVAL_FEATURE_CACHE=0`으로 끈다.
2장(보라매)은 최대 8 GiB이다.

### evaluate CPU 병렬

evaluate는 지표·곡선·bootstrap 계산을 여러 CPU 프로세스로 나눠 돈다. `site.yaml`의 `workers`가 프로세스 수이고
(기본 CPU 수와 8 중 작은 값), 환경변수 `MIVAL_WORKERS`로 덮어쓸 수 있다. 결과 값과 config_hash는 workers 값과
상관없이 같다. 프로세스마다 메모리를 따로 쓰므로 메모리가 모자라면 줄인다 (`workers: 1`이면 한 프로세스로 돈다).
참조 서버에서 `workers: 8`일 때 전체 메모리는 최대 2~3 GB였다(lvef 3.1 GB).

### xECG 커널 (Linux)

xECG의 sLSTM은 CUDA 커널을 쓴다. 번들의 `cache/torch_extensions`에는 **미리 컴파일된 커널**이 있는데,
번들을 만들 때와 **같은 절대 경로**에 풀었을 때만 그대로 쓰인다. 다른 경로에 풀면 첫 실행에서
다시 컴파일하므로 `nvcc`(CUDA toolkit)와 **gcc 14 이하**가 필요하다. 없으면 models [mival-xecg]가 실패한다.
Windows에서는 위 5절처럼 컴파일 없는 구현으로 돌므로 이 조건이 필요 없다.

### 소요 시간

참조 서버(NVIDIA L4 1장, vCPU 16개, RAM 60 GB)에서 MIMIC-IV-ECG로 잰 값이다(2026-10-19~20). linear probe feature 캐시를 켜고,
evaluate는 `workers: 8`로 돌렸다(`workers: 1`이면 evaluate가 5~6.5배 느리다. 예: potassium 41분, lvef 1시간 57분).
test ECG 수는 lvef 36,025건 등 코호트에 따라 다르다.

| study | retrieve | profile | preprocess | describe | models | evaluate | 합계 |
|---|---|---|---|---|---|---|---|
| lvef | 1분 | 1초 미만 | 23분 | 5초 | 44분 | 18분 | 1시간 26분 |
| ecg-age | 28초 | 1초 | 18분 | 6초 | 15분 | 6분 | 39분 |
| mortality | 29초 | 1초 | 15분 | 5초 | 3분 | 10분 | 29분 |
| potassium | 17분 | 1초 | 12분 | 6초 | 50분 | 7분 | 1시간 27분 |
| export | | | | | | | study당 4~12초 (4개 한 zip 21초) |

기관 서버의 시간은 코호트 크기, DICOM 읽기 속도, GPU 수에 따라 달라진다. CPU만 쓰는 서버는 측정하지 않았고 models 단계가 훨씬 오래 걸린다.

## 7. 결과 확인과 회신

`runs/<study>/_run/run.log` 마지막 줄에 evaluate 결과 폴더가 찍힌다.

```
runs/<study>/
├── retrieve/<hash>/   artifacts/: cohort_index.parquet, retrieve_summary.json, 실행한 SQL
├── profile/<hash>/    cohort_split.parquet (dev/test), profile_summary.json
├── preprocess/<hash>/ 모델 규격으로 변환된 파형
├── describe/<hash>/   Table 1, 제외·STARD 집계
├── models/<hash>/     predictions/*.parquet (ECG별 예측)
├── evaluate/<hash>/   metrics_long.parquet, comparisons.parquet, figures/*.png
├── exclusions/        제외된 ECG와 사유 (단계별)
└── _run/              단계별 로그, runs.json
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

### 연구실로 회신할 것: zip 하나

4개 study가 끝나면 `export_results.sh`(Windows는 `.cmd`)를 실행한다. `$MIVAL_HOME`에
`results-<site>-<YYMMDD>.zip` 하나가 만들어지고, **회신하는 파일은 이 zip 하나뿐이다**. `runs/` 폴더를 통째로 보내지 않는다.

```bash
scripts/export_results.sh                       # runs.json이 있는 study 전부
scripts/export_results.sh lvef ecg-age          # study를 골라서
scripts/export_results.sh --allow-small-stage-totals    # 아래 설명
```

zip에는 집계 값만 들어 있다: Table 1, 지표와 95% CI, 곡선 값, 제외·STARD 집계, 다시 그린 그림, 실행 기록(manifest, summary),
TRIPOD+AI 체크리스트와 방법 초안. 허용 목록에 있는 파일만 만들어지고, 다음은 **어떤 경우에도 나가지 않는다**:
`cohort_index`, `cohort_split`, 변환된 파형, `predictions`, `person_id`·`image_occurrence_id`·`local_path` 같은
식별 열, 개인 단위 날짜, 절대 경로. 만든 뒤 모든 파일을 다시 읽어 이런 열·값이 있으면 zip을 쓰지 않고 멈춘다.

- **작은 칸 가림.** study의 `min_cell`(10)보다 작은 칸(1~9)은 `masked`로 나간다. 합계와 나머지 칸에서 가려진 값을
  역산할 수 있으면 다른 칸도 함께 가린다 (선형계획으로 검사한 보완 가림). 가린 칸에서 계산한 비율·곡선 점은 NaN이다.
- **`--allow-small-stage-totals`.** 어떤 단계에서 제외된 ECG가 합쳐서 1~9건이면, 그 합은 단계 전후 건수에서
  역산되므로 가릴 수 없다. 기본은 이때 export를 거부한다. 기관이 그 건수를 내보내도 된다고 정한 뒤에만 이 옵션을
  쓰며, 쓴 사실은 zip의 `export_manifest.json`과 README에 적힌다.
- **zip 안의 README에 반출 전 검토 포인트가 있다.** 보내기 전에 기관 담당자가 이 README를 읽고 따른다. 끝나면 화면에도
  같은 안내가 한 줄 나온다.

## 8. 전체 파일 목록

### 저장소 (`$MIVAL_HOME/miva/`)

```
README.md                         이 문서
pyproject.toml                    패키지 정의 (mival), 의존성
.gitignore
config/
  micdm.env.example               DB 접속 파일 양식 (4.1절)
  site.yaml.example               기관 설정 양식: site, gpus, DBMS, schema, DICOM 경로 (4.2절)
sql/
  check_cohort.postgresql.sql     실행 전 코호트 규모 확인 쿼리, PostgreSQL (파이프라인은 쓰지 않음)
  check_cohort.sqlserver.sql      같은 쿼리, SQL Server
scripts/
  run_study.sh / run_study.cmd    study 하나를 6단계로 실행, Linux·Windows (6절). 인터프리터만 찾고 run_study.py를 부른다
  run_study.py                    6단계 실행 본체
  run_models.py                   models 단계: arm을 gpus 슬롯에 나눠 실행
  export_results.sh / .cmd        회신 zip 만들기 (7절). export_results.py를 부른다
  export_results.py               반출 본체
  check_bundle.py                 기관 설정과 카드의 가중치·sha256·코드·env·GPU·Windows 경로 길이 확인 (5절)
docs/
  study-yaml-reference.md         study.yaml 항목별 설명

studies/                          ── study.yaml: 무엇을 어떻게 평가할지
  lvef/study.yaml                 LVEF 분류(≤40)·회귀, 모델 4개 6 arm
  ecg-age/study.yaml              ECG 나이 회귀, 모델 2개
  mortality/study.yaml            1년 사망 생존분석, 모델 2개
  potassium/study.yaml            혈청 칼륨 회귀·판별 (K<3.5, K>5.5), 모델 2개

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
    mival_xecg_vanilla.py           xECG 가중치를 Windows용 컴파일 없는 sLSTM 배치로 변환 (5절)

src/mival/                        ── 엔진 (모델·기관 이름이 코드에 없다)
  __init__.py
  modelcard.py                    카드 읽기, ${MIVAL_HOME} 치환
  placeholders.py                 ${NAME} → 환경변수 값 (카드, site.yaml 공용)
  runtime_platform.py             카드의 플랫폼별 실행 설정 적용 (Windows의 xECG 등)
  contract.py                     입력 규격과 제외 사유 어휘
  compiler.py                     입력 규격 → 전처리 recipe
  ops.py                          전처리 연산 (lead 선택, resample, crop/pad, 단위, scaling)
  signal.py                       ECG 메모리 표현
  criteria.py                     study.yaml의 임상·통계 기준(나이 구간, 회귀 cut, min_cell 등) 읽기
  suppression.py                  작은 칸 가림 (보완 가림, 선형계획 검사)
  export.py                       회신 zip 만들기: 허용 목록, 가림, 재검사
  reporting.py                    TRIPOD+AI 체크리스트·방법 초안
  reporting_tripod_ai.yaml        체크리스트 항목
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
    _feature_cache.py               linear probe feature 캐시 (6절)
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
    curves.py                       고정 격자 곡선과 bootstrap 띠
    tiers.py                        지표의 보고 등급 (core / descriptive / supplementary)
    bootstrap.py                    사람 단위 bootstrap CI, 모델 비교
  pipeline/                       실행 틀
    __init__.py
    cli.py                          `mival run <stage>` 명령
    study.py                        study.yaml 읽기, site.yaml 합치기
    stage.py                        단계 등록과 실행 (끝난 단계 건너뛰기)
    parallel.py                     evaluate 작업을 여러 CPU 프로세스로 나눠 돌리기 (site.yaml workers)
    layout.py                       결과 폴더 구조
    envs.py                         env별 인터프리터 찾기 (Linux bin/, Windows python.exe)
    manifest.py                     실행 기록 (입력, 설정, 코드, env)
    ledger.py                       제외 기록
    hashing.py                      sha256, config_hash
    runkey.py                       결과 좌표 (model, label, split, subgroup …)
    tables.py                       parquet 입출력
  stages/                         6단계 구현
    __init__.py
    retrieve.py                     MI-CDM(PostgreSQL/SQL Server)에서 ECG·라벨 추출, DICOM 경로 연결
    profile.py                      사람 단위 dev/test 분할, 양성 수 확인
    preprocess.py                   카드별 파형 변환
    describe.py                     Table 1, 제외·STARD 집계
    models.py                       arm별 예측, linear probe 학습
    evaluate.py                     test 지표, CI, 그림
    _predictions.py                 예측 읽기 (evaluate 공용)
    _comparisons.py                 짝 지은 모델 비교 (evaluate 보조)
    _curve_rows.py                  곡선 행 만들기 (evaluate, export 공용)
    misclassify.py                  오분류 사례 분석 (파이프라인에서 제외됨, 실행되지 않음)
    sql/
      retrieve_ecg_label.sql          ECG + 가장 가까운 measurement 라벨 (PostgreSQL)
      retrieve_ecg_label.sqlserver.sql  같은 쿼리, SQL Server (outer apply, datediff)
      retrieve_ecg_person.sql         ECG + 출생연도·사망·마지막 방문 (나이·사망 라벨, 두 DBMS 공용)
      label_snapshot.sql              라벨 원천 스냅샷

```

### 기관이 만드는 것

```
$MIVAL_HOME/config/site.yaml      기관 설정 (4.2절)
$MIVAL_HOME/config/micdm.env      DB 접속 (4.1절)
$MIVAL_HOME/results-<site>-<YYMMDD>.zip   export_results가 만드는 회신 파일 (7절)
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
| models [mival-xecg] 실패 | GPU, 드라이버. GPU가 없으면 정상적으로 건너뛴다. Linux에서 번들 제작 때와 다른 경로에 풀었다면 커널을 다시 컴파일하므로 `nvcc`와 gcc 14 이하가 필요하다 (6절) |
| Windows에서 `check_bundle`에 `runtime.windows override applies` 줄이 나옴 | 정상이다 (5절). 문제로 세지 않는다 |
| Windows에서 `check_bundle`이 경로 260자를 지적 | `LongPathsEnabled`를 켜거나 `C:\mival` 같은 짧은 경로에 설치 (5절) |
| 서버 메모리 부족 (models 단계) | `MIVAL_FEATURE_CACHE_MAX_GB`를 줄이거나 `MIVAL_FEATURE_CACHE=0` (6절) |
| export가 "한 단계에서 1~9건 제외"로 거부됨 | 그 건수를 내보내도 되는지 기관이 정한 뒤 `--allow-small-stage-totals` (7절) |
| export가 식별 열·경로·날짜를 찾아 실패 | zip은 만들어지지 않는다. 메시지를 연구실에 알린다 |
