# Model wrappers

카드의 `builder`가 가리키는 작은 torch/keras 모듈이다. 저자 코드가 하는 전처리나 출력 처리 중
입력 규격(input_contract)으로 표현할 수 없는 것(checkpoint에 저장된 정규화 통계, 정규화 뒤의
padding, dict를 돌려주는 forward 등)을 여기서 한다.

| 파일 | 카드 |
|---|---|
| `mival_wrap_cavalab.py` | `cavalab-deepsurv-code15`, `cavalab-mtlr-code15` |
| `mival_wrap_potassium.py` | `kardionet-k-12lead`, `vonbachmann-k` |
| `mival_wrap_singstad.py` | `singstad-ecg-age` |
| `mival_xecg_vanilla.py` | `xecg`의 `runtime.windows.state_dict_transform` (builder가 아님) |

저자 코드는 `$MIVAL_HOME/code/<repo>`에서 읽는다 (`MIVAL_HOME`이 없으면 `/opt/<repo>`).
여러 카드의 `code_path`가 한 `sys.path`를 공유하므로 파일마다 모듈 이름이 다르다.

`mival_xecg_vanilla.py`는 builder가 아니다. xECG 체크포인트는 sLSTM `cuda` backend의 파라미터 배치로
저장되어 있고, Windows에서 쓰는 `vanilla` backend는 배치가 다르다. torch adapter가 strict load 직전에
이 함수로 recurrent kernel(`permute(0, 2, 1)`)과 bias(head·gate 순서 → gate·head 순서)를 바꾼다.
카드는 `${MIVAL_HOME}/miva/registry/wrappers`(이 디렉터리가 번들에 실린 자리)에서 읽는다.
