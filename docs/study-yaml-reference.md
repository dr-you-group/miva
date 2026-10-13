# study.yaml 항목 설명

`studies/<task>/study.yaml`과 `config/site.yaml`의 각 항목이 무엇인지 한두 줄로 설명한다. 이 저장소의 4개 study가 쓰지 않는 항목도 함께 적혀 있다.

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
| `label_concept_id` | `measurement`일 때 라벨로 쓸 CDM measurement concept (예: 칼륨 3023103). |
| `window_days` | ECG 날짜 ± 며칠 안의 검사값 중 가장 가까운 것을 라벨로 쓴다. 0이면 같은 날. |
| `window_days_sens2` | 민감도 분석용으로 넓힌 window(일). |
| `implausible_below` | 라벨 값이 이 값 이하이면 비정상으로 보고 제외한다 (라벨 단위). 생략하면 5, `null`이면 이 기준으로 제외하지 않는다. |
| `primary_cutoff` | 분류 라벨 기준. 값 ≤ cutoff이면 양성(1). 예: LVEF ≤ 40. |
| `sens1_cutoff` | 민감도 분석용 분류 기준 (예: LVEF ≤ 50). |

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

## 5. evaluate: test에서 지표를 계산한다

평가 지표 자체는 `label_def`에 따라 자동으로 정해진다. 아래 항목은 계산 기준만 바꾼다.

| 항목 | 설명 |
|---|---|
| `outcomes` | 결과 이름. 결과표의 outcome 축에 표시된다. |
| `regression_cuts` | 회귀 값을 이진으로 볼 기준값. 기준마다 "값 ≤ cut" 판별 AUROC를 낸다. `label_def`별로 따로 줄 수 있다. |
| `horizons_days` | 생존 모델의 AUROC를 볼 시점(일). |
| `threshold_policy` | 분류 모델의 운영 임계값을 정하는 방식. dev에서 정하고 test에 적용한다. `refit_sens95`(민감도 목표 지점), `refit_youden`(Youden 지수 최대), `legacy`(고정값 `legacy_threshold`, 기본 0.5). |
| `sensitivity_target` | `refit_sens95`일 때 목표 민감도. dev에서 민감도가 이 값(기본 0.95)이 되는 임계값을 찾고, 그 임계값으로 test의 sensitivity·specificity·PPV·NPV를 계산한다. |
| `decision_thresholds` | net benefit(decision curve)을 계산할 위험 기준값들. |
| `subgroups` | 하위군(성별, 나이대 등)별로 지표를 반복 계산할 컬럼. attributes 입력이 필요하다. |
| `min_events` | 하위군 사건 수가 이보다 적으면 값을 숨긴다. |
| `bootstrap_replicates` | 95% CI를 위한 사람 단위 bootstrap 반복 수. 0이면 CI를 내지 않는다. |
| `bootstrap_alpha` | CI 수준. 0.05면 95% CI. |
| `figures` | ROC, calibration 등 그림을 만들지. |
