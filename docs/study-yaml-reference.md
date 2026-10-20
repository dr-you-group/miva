# study.yaml 항목 설명

`study-schema-tobe.compact.yaml`의 각 항목이 무엇인지 한두 줄로 설명한다.

## 최상위

| 항목 | 설명 |
|---|---|
| `study_id` | study 이름. 결과 폴더 경로와 실행 기록(run key)에 쓰인다. |
| `site` | 기관(데이터 소스) 이름. 결과표에서 기관을 구분하는 축이다. |
| `seed` | 무작위 요소(1인 1건 추출, dev/test 분할, bootstrap)에 공통으로 쓰는 seed. 같은 seed면 같은 결과가 나온다. |

## 1. retrieve: ECG를 찾고 라벨을 붙인다

| 항목 | 설명 |
|---|---|
| `dbms` | `postgresql`(기본) 또는 `sqlserver`. 라벨 쿼리를 DBMS에 맞는 것으로 고른다. |
| `dsn_env` | DB 접속 정보가 든 env 파일 경로. `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` (PostgreSQL은 `PGHOST` 등도 됨). |
| `schema` | OMOP CDM이 들어 있는 DB schema 이름. SQL Server는 `database.schema` 형식도 된다 (예: `CDM2025.dbo`). |
| `modality_concept_id` | `image_occurrence`에서 ECG만 고르는 concept. 12-lead ECG는 4145308 (CXR이 같은 테이블에 섞여 있어 필요). |
| `local_path_root` | `null`(또는 생략)이면 CDM `image_occurrence.local_path`를 그대로 DICOM 경로로 쓴다. 경로를 주면 local_path의 `/files/` 뒤 부분을 이 루트의 `files/` 아래로 다시 연결한다 (CDM을 만든 곳과 파일 위치가 다를 때). |
| `require_local_file` | true면 DICOM 파일이 실제로 없는 ECG를 제외한다. |
| `ecg_selection.per_person` | `all`은 조건을 통과한 ECG를 전부 쓰고(1인 다건), `one`은 사람마다 1건만 쓴다. 생략하면 `all`. |
| `ecg_selection.rule` | `one`일 때 고르는 기준. `first`(가장 이른 ECG), `last`(가장 늦은 ECG), `nearest_label`(라벨 측정과 가장 가까운 ECG), `random`(seed 무작위). 같은 날 ECG는 image_occurrence_id 순서로 정한다(CDM에 시각 없음). |
| `label_source.kind` | 라벨을 만드는 방식. `measurement`(CDM 검사값), `measurement_file`(외부 파일의 검사값), `age_at_ecg`(ECG 시점 나이), `death_within`(N일 내 사망), `machine_measurement`(장비 측정 PR/QRS/QT). |
| `label_source.path` | `measurement_file`, `machine_measurement`일 때 읽을 파일 경로. |
| `label_source.horizon_days` | `death_within`일 때 사망을 볼 기간(일). |
| `label_source.follow_up_after_last_visit_days` | `death_within`일 때 마지막 방문 뒤 이 일수까지만 사망이 기록된다고 본다. 그 뒤는 관찰되지 않은 것으로 검열한다. 생략하면 365. 반출 묶음의 `export_manifest.json`(criteria)에 쓴 값과 출처(study.yaml 또는 engine default)가 남는다. |
| `label_concept_id` | `measurement`일 때 **필수**. 라벨로 쓸 CDM measurement concept (예: 칼륨 3023103). 기본값이 없고, 빠지면 `retrieve.label_concept_id is required` 오류가 난다. |
| `window_days` | ECG 날짜 ± 며칠 안의 검사값 중 가장 가까운 것을 라벨로 쓴다. 0이면 같은 날. |
| `window_days_sens2` | 민감도 분석용으로 넓힌 window(일). |
| `implausible_below` | 라벨 값이 이 값 이하이면 비정상으로 보고 제외한다 (라벨 단위). 생략하면 5, `null`이면 이 기준으로 제외하지 않는다. |
| `primary_cutoff` | 분류 라벨 기준. 값 ≤ cutoff이면 양성(1). 예: LVEF ≤ 40. 기본값이 없다. 생략하면 `label_primary`와 `label_sens2`는 NaN이다. |
| `sens1_cutoff` | 민감도 분석용 분류 기준 (예: LVEF ≤ 50). 기본값이 없다. 생략하면 `label_sens1`은 NaN이다. 두 cutoff는 서로 독립이고, 둘 다 없으면 retrieve_summary.json에 `positives`가 없다. |

## 2. profile: 사람 단위로 dev / test를 나눈다

| 항목 | 설명 |
|---|---|
| `test_fraction` | test로 뗄 사람 비율. 나머지는 dev. |
| `n_folds` | dev 안의 교차검증 fold 수. probe 학습과 하이퍼파라미터 선택에 쓴다. |
| `stratify_on` | 분할할 때 비율을 맞출 라벨 컬럼. 연속값·생존 task는 `none`. |
| `min_test_positives` | test 양성 수가 이보다 적으면 실패로 멈춘다. 연속값 task는 0. |

## 3. preprocess: 파형을 모델 입력 규격에 맞춘다

| 항목 | 설명 |
|---|---|
| `loader` | 파형을 읽는 방식. `dicom`(실데이터) 또는 `npz`(테스트용). |
| `registry` | 모델 카드(json) 폴더. 카드마다 적힌 입력 규격(샘플링, 길이, 단위)에 맞춰 변환한다. |
| `allow_upsample` | 카드 규격보다 샘플링이 낮은 ECG를 올려서 맞추는 것을 허용할지. false면 제외. |
| `pad_policy` | 길이가 모자랄 때 `reject`(제외) 또는 `zero`(0으로 채움). |
| `perturbation.mode` | 강건성 실험 조합 방식. `ofat`은 한 번에 한 축만 바꾸고, `cartesian`은 모든 조합. |
| `perturbation.axes.resample` | 샘플링 주파수 수준(Hz). |
| `perturbation.axes.duration` | 신호 길이 수준(초). |
| `perturbation.axes.lead_dropout` | lead를 빼는 조건 (`drop_V3V4`, `precordial_only`, `limb_only`). |
| `perturbation.axes.amplitude_scale` | 진폭 배율. |
| `perturbation.axes.noise` | 잡음 종류 (`baseline_wander`, `emg`, `powerline_<Hz>hz`). |

## 3b. describe: 속성표, Table 1, 제외 흐름을 만든다

preprocess 다음에 돈다. 입력은 `cohort_index`, `cohort_split`, `ecg_attributes`, `source_index`이고, 선택 입력 `runs`(runs.json)를 주면 그 실행의 제외 기록만 센다. 주지 않으면 ledger 전체를 센다. 아무것도 제외하지 않고 작은 셀도 가리지 않는다(가림은 export).

출력은 `attributes.parquet`(cohort ECG마다 나이, 나이대, 성별, 연도, 장비, 샘플링, 길이, 리드 수), `table1_long.csv`(열 `overall`, `dev`, `test`, outcome이 있으면 `test:<outcome>=1`, `test:<outcome>=0`), `exclusions_long.csv`(단계, 사유별 제외 수를 전체, 성별, 나이대로)이다. 통계에 최솟값, 최댓값은 없다.

모든 항목이 선택이다. 없으면 기본값을 쓴다.

| 항목 | 설명 |
|---|---|
| `age_bands` | 나이대 하한 경계 목록. 기본 `[50, 65, 80]`이고 라벨은 `<50`, `50-64`, `65-79`, `>=80`. 증가 순서여야 하고 비어 있으면 오류. |
| `outcome` | 결과 라벨인 `cohort_index` 열 이름. 분류는 `label_primary`, 생존은 `label_event`. 주면 모든 열에 이 라벨의 1/0 분포(수, %)가 들어가고(분모는 라벨이 있는 ECG 수), test 열을 결과 유무(1/0)로도 나눈다. 없으면 둘 다 하지 않고, cohort_index에 없는 열이면 오류. |
| `conditions` | 예약. 기본 `[]`. v1은 condition_occurrence를 읽지 않으므로 비어 있지 않으면 오류. |

## 4. models: 모델별로 예측한다

| 항목 | 설명 |
|---|---|
| `registry` | 모델 카드 폴더. preprocess와 같게 둔다. |
| `cohort_sources` | 이 데이터의 코호트 이름. 카드의 학습 데이터와 겹치는지(오염) 판정할 때 쓴다. |
| `batch_size` | 추론 batch 크기. |
| `arms[].model_id` | 돌릴 모델 (카드의 model_id). |
| `arms[].training_mode` | `inference_only`(공개 가중치 그대로), `linear_probe`(backbone 고정, head만 dev로 학습), `partial_unfreeze`, `full_finetune`. |
| `arms[].label_def` | 어떤 라벨로 평가할지. 이 값이 평가 유형을 정한다: `primary`/`sens*`는 분류, `value`는 회귀, `pr`/`qrs`/`qt`는 분할 간격, `survival`은 생존. |
| `arms[].hparam_grid` | 학습 모드에서 dev fold로 고를 하이퍼파라미터 후보 (예: lr). |
| `arms[].hparams` | 학습 모드에서 고정할 하이퍼파라미터. |
| `arms[].unfreeze_groups` | `partial_unfreeze`일 때 풀어서 학습할 층 묶음. |
| `arms[].early_stopping_fold` | 조기 종료 판단에 쓸 dev fold 번호. |

models 단계를 띄우는 실행 스크립트(`run_study.py`, `run_models.py`)는 arm 하나를 job 하나로 돌린다. 이때 아래 값을 읽는다. 어느 것도 config_hash에 들어가지 않는다(카드 파일 자체는 들어간다).

| 위치 | 항목 | 설명 |
|---|---|---|
| 카드 `x-mival.runtime` | `env` | 이 모델을 돌릴 env 폴더 (반입 번들의 python-build-standalone 폴더, 또는 conda env). 그 안의 `bin/python`(Linux) 또는 `python.exe`(Windows)를 쓴다. |
| 카드 `x-mival.runtime` | `requires_gpu` | `true`면 GPU가 없을 때 그 arm을 돌리지 않고 사유를 남긴다. 기본 `false`. |
| 카드 `x-mival.runtime` | `run_env` | 이 모델의 job에만 넣을 환경변수 (예: CUDA 확장 캐시 `TORCH_EXTENSIONS_DIR`). `${MIVAL_HOME}`을 쓸 수 있다. |
| 카드 `x-mival.runtime` | `compile_env` | CUDA 확장을 컴파일한 환경의 기록. 실행 때 넣지 않는다. 이 키가 있는 카드는 Windows에서 `runtime.windows`가 없으면 check_bundle이 문제로 센다. |
| 카드 `x-mival.runtime` | `prebuilt_torch_extensions` | `true`면 이 카드의 모듈을 만드는 동안(torch adapter `load`) 두 가지를 한다. (1) 코드가 `torch.utils.cpp_extension.load(name, ...)`를 부를 때 `$TORCH_EXTENSIONS_DIR/<name>/<name>.so`(Windows `.pyd`)가 있으면 그 파일을 torch가 빌드 끝에 하듯 import해서 돌려준다. torch의 ninja 재빌드 확인을 건너뛰므로 nvcc, 컴파일러, CUDA 헤더, 빌드 때의 환경변수와 설치 경로가 달라도 된다. 파일이 없으면 torch의 `load`를 그대로 부른다. (2) torch가 CUDA 툴킷을 못 찾았으면(`CUDA_HOME` 없음) 그동안 `cpp_extension.CUDA_HOME`을 자리표시 경로로 둔다. import할 때 include 경로만 묻고 컴파일은 하지 않는 코드(예: 컴파일하지 않는 backend)가 멈추지 않게 하려는 것이고, (1)의 fallback 빌드에는 자리표시를 보이지 않는다. 불러온 확장 이름·파일·sha256과 자리표시 사용 여부는 `train_log.jsonl`의 `model_card.prebuilt_torch_extensions`에, 확장 이름은 `export_manifest.json` `arms`의 `prebuilt_torch_extensions`에 남는다. 파일은 이름으로만 고르므로 같은 env·소스로 빌드한 것이어야 한다(번들은 `scripts/bundle/build_kernels.py`로 같은 env에서 만든다). 확장 폴더는 환경변수 `TORCH_EXTENSIONS_DIR`에서만 읽는다(run_models가 카드 `run_env`로 넣는다). 그 변수 없이 카드를 올리는 프로세스(직접 쓴 스크립트 등)는 torch 빌드로 간다. 카드의 코드가 `load`를 이름으로 import한다면 그 import가 모델을 만드는 동안 처음 일어나야 한다(arm 프로세스에서는 그렇다). 그 전에 import되어 있으면 torch의 `load`가 쓰이고, 미리 빌드한 파일이 있는데 아무것도 불러오지 않았으면 warning과 기록(`load_bound_before`)에 그 모듈 이름이 남는다. |
| 카드 `x-mival.runtime` | `windows` | Windows(`sys.platform == "win32"`)에서만 카드를 읽을 때 적용하는 덮어쓰기. 예: `{"builder_kwargs": {"config": {"backend": "vanilla"}}, "compile_env": null, "slstm_backend": "vanilla"}`. `builder_kwargs`는 `x-mival.builder.kwargs`에 깊게 합친다(중첩 객체는 키별로 합치고, 나머지 값은 `null`까지 그대로 바꾼다). 그 밖의 키는 `runtime`의 같은 키를 바꾸고, `null`이면 그 키를 지운다(`compile_env: null`: 이 플랫폼에서는 컴파일하지 않는다). 다른 플랫폼에서는 아무것도 바꾸지 않는다. 적용되면 models 단계 manifest의 warnings와 `train_log.jsonl`의 `model_card.runtime_override`(이름·플랫폼·바뀐 항목)에 남고, `run_models.py` 결과 JSON의 arm 항목과 반출 묶음 `export_manifest.json`의 `arms`(arm마다 `runtime_override`, `runtime_platform`)에 이름이 남는다. check_bundle은 문제가 아니라 info로 적는다. 적용된 덮어쓰기는 config_hash에 들어간다(적용되지 않으면 hash는 그대로). |
| 카드 `x-mival.runtime` | `state_dict_transform` | 체크포인트를 strict load하기 직전에 state dict를 바꾸는 함수. `"module:function"`이면 카드의 `code_path`에서, `{"function": "module:function", "code_path": "<dir>"}`이면 그 `code_path`에서 builder 모듈과 같은 방식으로 읽는다. 함수는 state dict를 받아 load할 state dict를 돌려준다. 보통 `runtime.windows` 안에만 둔다(예: xECG는 sLSTM `cuda` 배치로 저장된 체크포인트를 `vanilla` 배치로 바꾸는 `mival_xecg_vanilla:to_vanilla`, `${MIVAL_HOME}/miva/registry/wrappers`). override가 이것을 넣으면 `train_log.jsonl`의 `model_card.runtime_override.state_dict_transform`과 `export_manifest.json` `arms`의 `state_dict_transform`에 함수 이름만 남는다. check_bundle은 그 모듈 파일이 없으면 문제로 센다. |
| 환경변수 | `MIVAL_RUNTIME_PLATFORM` | 덮어쓰기를 고를 때 `sys.platform` 대신 쓸 값. `win32`이면 Linux에서도 `runtime.windows`가 적용된다. 시험과 기준 서버의 동등성 비교(`scripts/dev/compare_runtime_override.py`)용이다. |
| `config/site.yaml` | `gpus` | 쓸 GPU 번호 목록. 동시에 `len(gpus)`개 arm이 돌고 각 job은 `CUDA_VISIBLE_DEVICES`로 GPU 하나만 본다. 비우면 GPU가 있을 때 `[0]`, 없으면 CPU에서 하나씩. 번호는 이 프로세스가 보는 GPU 순서다. 실행 전에 `CUDA_VISIBLE_DEVICES=4,5`처럼 가려 두었으면 `0`은 4번, `1`은 5번 GPU가 된다. 동시에 도는 arm마다 feature cache(아래)가 따로 잡히므로, 기본 4 GiB면 `gpus`가 8개일 때 RAM을 최대 32 GiB 더 쓴다. |
| `config/site.yaml` | `workers` | evaluate 단계가 slice(run_key × outcome × 하위군, dev로 맞춘 update)와 자동 arm 쌍 비교를 나눠 돌릴 CPU 프로세스 수. 비우면 CPU 수와 8 중 작은 값, `1`이면 한 프로세스에서 차례로. 환경변수 `MIVAL_WORKERS`가 있으면 그 값이 이긴다. 양의 정수가 아니면 evaluate가 시작할 때 멈춘다. 프로세스는 Windows와 같은 `spawn`으로 띄우고, 계산 단위마다 BLAS를 1 thread로 묶는다(`workers: 1`도 같다). 결과는 workers 값과 상관없이 같고 config_hash에도 들어가지 않는다. 쓴 값은 evaluate manifest의 `compute.workers`에 남는다. worker 하나는 pandas·scikit-learn을 올린 Python 프로세스라 대략 150–250 MB를 더 쓰고, 계산 중인 slice의 열 몇 개를 받는다. 메모리가 모자라면 줄인다. 두 실행의 표를 비교하려면 `scripts/dev/compare_evaluate_outputs.py <run_a> <run_b>`. |
| 환경변수 | `MIVAL_FEATURE_CACHE` | `linear_probe` arm이 고정 backbone의 출력을 후보·fold·최종 학습·예측 사이에 다시 쓴다. `0`(또는 `false`/`off`/`no`)이면 끈다. 결과가 아니라 속도만 바꾸므로 config_hash에 들어가지 않는다. GPU에서는 batch 구성이 달라져 마지막 자리 수치가 조금 다를 수 있다. 썼는지, 얼마나 다시 썼는지는 `train_log.jsonl`의 `training.feature_cache`에 남는다. |
| 환경변수 | `MIVAL_FEATURE_CACHE_MAX_GB` | arm 하나(job 하나)의 feature cache 상한(GiB). 기본 4. 상한에 닿으면 그 뒤 행은 계산만 하고 저장하지 않는다. |
| Windows | 경로 길이 | Windows는 경로 260자 제한이 있다(`LongPathsEnabled`가 꺼져 있으면). 가장 긴 파일은 예측 파일 `<MIVAL_RUNS>\<study>\models\<hash>\artifacts\predictions\<run_key>.parquet`이고 `C:\mival\runs` 아래에서 약 230–250자다. check_bundle이 `MIVAL_RUNS`와 study의 site·arm으로 이 길이를 어림해 259자를 넘으면 문제로 센다. 고치는 법: 레지스트리 `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`의 `LongPathsEnabled`를 1로 하고 재부팅하거나, `MIVAL_RUNS`를 `D:\mival\runs`처럼 짧은 곳에 둔다. |

## 5. evaluate: test에서 지표를 계산한다

평가 지표 자체는 `label_def`에 따라 자동으로 정해진다. 아래 항목은 계산 기준만 바꾼다.

| 항목 | 설명 |
|---|---|
| `outcomes` | 결과 이름. 결과표의 outcome 축에 표시된다. |
| `regression_cuts` | 회귀 값을 이진으로 볼 기준값. 숫자만 쓰면 "값 ≤ cut"이 사건이다. `{value, op}`로 쓰면 방향을 정한다(`op`는 `<`, `<=`, `>`, `>=`). 예: 칼륨 `[{value: 3.5, op: "<"}, {value: 5.5, op: ">"}]`. `label_def`별로 따로 줄 수 있다. 기본값이 없다. 생략하면 회귀 arm은 회귀 지표만 내고 cut 지표는 내지 않으며, `export_manifest.json`에 `not set: no cut metrics`로 남는다. 빈 목록이나 `null`은 오류다. |
| `net_benefit_grid` | decision curve의 임계 확률 격자 `{start, stop, step}`. 기본 `0.01, 0.99, 0.01`(99개 점). 모든 값이 0과 1 사이여야 한다. 이 격자에서 `curves_long.parquet`의 `threshold_grid` 곡선(TP/FP/TN/FN, sens/spec, PPV/NPV, net benefit, treat-all)을 낸다. 곡선은 test split에서, 사건·비사건이 모두 `min_events` 이상인 slice만 만든다. 비율과 net benefit에는 모든 slice에 bootstrap CI가 붙는다. `calibration` 곡선(예측 위험 0.01–0.99, 예측값 p1–p99 밖은 빈 값)의 CI는 subgroup `all`에만 낸다. 다른 subgroup은 점 곡선만 있고 CI는 빈 값이다. 실행 시간 때문이다. `risk_distribution`(20구간 사건·비사건 수)은 점 값이다. |
| `horizons_days` | 생존 모델의 AUROC를 볼 시점(일). |
| `threshold_policy` | 분류 모델의 운영 임계값을 정하는 방식. dev에서 정하고 test에 적용한다. `refit_sens95`(민감도 목표 지점), `refit_youden`(Youden 지수 최대), `legacy`(고정값 `legacy_threshold`, 기본 0.5). |
| `sensitivity_target` | `refit_sens95`일 때 목표 민감도. dev에서 민감도가 이 값(기본 0.95)이 되는 임계값을 찾고, 그 임계값으로 test의 sensitivity·specificity·PPV·NPV를 계산한다. |
| `decision_thresholds` | `metrics_long`에 `net_benefit@t` 행을 따로 낼 위험 기준값들. 기본은 없음이다. 적은 값에서만 행이 나온다. 전체 곡선은 `net_benefit_grid`에서 나온다. |
| `decision_band` | decision curve 그림에 칠할 임상적으로 그럴듯한 임계 구간 `[lo, hi]`. 기본은 없음이고, 없으면 칠하지 않는다. |
| `subgroups` | 하위군(성별, 나이대 등)별로 지표를 반복 계산할 컬럼. describe의 속성표 열 이름을 쓴다. |
| `min_events` | 하위군 사건 수가 이보다 적으면 값을 숨긴다. 기본 10. |
| `bootstrap_replicates` | 95% CI를 위한 사람 단위 bootstrap 반복 수. 0이면 CI를 내지 않는다. |
| `bootstrap_alpha` | CI 수준. 0.05면 95% CI. |
| `figures` | ROC, calibration 등 그림을 만들지. |

## 6. export: 반출 묶음

`export_results.sh`(Linux), `export_results.cmd`(Windows)가 만든다. 단계(stage)가 아니므로 아래 항목은 어느 config_hash에도 들어가지 않는다.

| 항목 | 설명 |
|---|---|
| `min_cell` | 반출 표에서 셀 수가 이보다 작으면 가린다. 기본 10, 1 이상. |
| `prespecified_subgroups` | 프로토콜이 미리 정한 하위군(예: `[sex, age_band]`). Methods 초안이 `evaluate.subgroups` 중 이것은 사전 지정, 나머지는 탐색적이라고 쓴다. 생략하면 "[prespecified or exploratory: fill in]"으로 남긴다. |

실행할 때 주는 것:

| 위치 | 항목 | 설명 |
|---|---|---|
| 명령줄 | `--allow-small-stage-totals` | 한 단계의 제외 수 합계가 1–`min_cell − 1`건이면(단계 앞뒤 인원의 차로 그 수를 알 수 있다) 반출을 멈춘다. 기관이 그 수를 내보내도 된다고 정한 뒤에만 이 옵션을 준다. `export_manifest.json`과 README에 남는다. study 이름 앞이나 뒤 어디에 써도 된다. 예: `export_results.sh --allow-small-stage-totals lvef`. |
| 환경변수 | `MIVAL_RUNS` | 실행 결과가 있는 곳(기본 `$MIVAL_HOME/runs`). 실행 뒤 결과 폴더를 옮겼으면(다른 디스크, 다른 드라이브 문자) 옮긴 곳을 준다. runs.json과 manifest에 적힌 옛 경로는 지금의 `MIVAL_RUNS`·`MIVAL_HOME` 아래로 다시 읽고, evaluate 입력은 기록된 sha256과 같아야 한다. |

Table 1의 `unit` 열은 그 행이 무엇을 세는지(`person` 또는 `ECG`) 적는다. `ecg_selection.per_person: all`이면 한 사람이 ECG 여러 건으로 들어가므로 나이 등 요약은 ECG 단위다. 사람 수는 `persons` 행에 있다.

Table 1의 연속형 요약(평균, SD, 분위수)은 그 값의 개수가 `min_cell`보다 적은 열에서 비운다. 또 열들이 더해지는 관계(overall = dev + test, test = 결과군 열 + 라벨 없는 test ECG)에서 어느 한 부분이 1–`min_cell − 1`개이면, `n × 평균`의 차로 그 부분의 평균을 알 수 있으므로 그 변수의 연속형 요약을 모든 열에서 비운다.
