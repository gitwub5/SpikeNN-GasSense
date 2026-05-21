# Gas Sensor Classification with Spiking Neural Networks (SNN)

이 문서는 UCI "Gas Sensor Array Drift Dataset" 데이터셋을 바탕으로, **센서의 반응을 통해 6가지 가스의 종류를 분류(Classification)**하는 SNN (Spiking Neural Network) 파이프라인의 전체 구조와 과정을 설명합니다. 

특히 시간이 지남에 따라 센서 반응이 미세하게 틀어지는 **Drift 현상**을 고려하여, 초기 수집 데이터로 학습하고 나중 수집 데이터로 테스트하는 시나리오를 채택하고 있습니다.

> **NPX Trainer 호환**: 본 파이프라인은 `.npx-xarvis/npx_trainer`와 동일한 학습 방식을 사용합니다. 이를 통해 추후 npx_trainer에 데이터셋만 추가하면 RISC-V RTL 생성까지 연결할 수 있습니다.

---

## 1. 개요 (Overview)

본 파이프라인의 핵심 구조는 다음과 같습니다.
1. **입력:** 16개의 연속(Time-Series) 반응에서 사전에 추출 및 통계 요약된 **128차원의 정적 피처(Static Feature) 벡터**
2. **출력:** Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene 를 분별하는 6개 클래스 확률(Spike Count)
3. **가스 인코딩 방식:** Constant Current Input (지속적인 입력 주입)을 통해 내부적으로 스파이크 발화를 유도
4. **모델 구조:** `Linear → LIF → Linear → LIF → Linear → LIF` 구조의 3-Layer SNN

> **왜 정적 데이터에 SNN을 사용하고 Time Step이 들어가나요?**
> SNN은 시간에 따른 스파이크(1, 0)만 처리합니다. 외부에서 원본 128차원 데이터를 수많은 0과 1로 미리 쪼개는(Rate Coding) 대신, 모델의 첫 번째 선형 계층에 **매 Time Step 마다 동일한 128차원 실수값을 상수 전류처럼 지속적으로 흘려넣습니다.** 
> 입력 값이 큰 뉴런은 전압이 빨리 차올라 스파이크를 자주 방출하고, 작은 값은 드물게 방출하는 자연스러운 인코딩이 모델 내부 공간에서 이루어집니다. 

---

## 2. 파이프라인 파일 및 구조

```text
spike_nn/
├── data_raw/drift_gas_sensor/        <-- 원본 수집 데이터 (batch1.dat ~ batch10.dat)
├── dataset/
│   └── drift_gas_dataset.py          <-- PyTorch DataLoader, Batch 결합 및 표준화 코드
├── train/
│   └── train_drift_gas.py            <-- Classification용 SNN 모델 정의 및 메인 학습 모듈
├── utils/
│   └── drift_gas_sensor/
│       ├── visualize_drift_gas.py    <-- PCA 시각화 및 피처 평균 그래프 생성 스크립트
│       └── visualize_drift_prediction.py  <-- 학습된 모델의 Batch별 정확도 시각화
├── analyze/drift_gas_sensor/         <-- 시각화 결과 저장소
├── model_weights/drift_gas_sensor/   <-- 학습된 모델 가중치 저장소
└── npx_drift_gas_data.py             <-- NPX Trainer 통합용 데이터셋 모듈
```

---

## 3. 모델 아키텍처

### 3-1. 네트워크 구조 (`GasSensorSNN`)

```text
Input [batch, 128]
     │
     ├─ fc1: Linear(128 → 256, bias=False)
     ├─ lif1: Leaky(β=0.95, reset=subtract, init_hidden=True)
     │
     ├─ fc2: Linear(256 → 128, bias=False)
     ├─ lif2: Leaky(β=0.95, reset=subtract, init_hidden=True)
     │
     ├─ fc3: Linear(128 → 6, bias=False)
     └─ lif3: Leaky(β=0.95, reset=subtract, init_hidden=True)
           │
      Output: spike (6 classes)
```

### 3-2. NPX Trainer 호환 설계 포인트

| 항목 | 설정값 | 이유 |
|------|--------|------|
| `init_hidden` | `True` | 뉴런이 내부적으로 membrane potential 관리. NPX의 `NpxModule.forward()`와 동일 패턴 |
| `spike_grad` | `None` | snntorch 기본 surrogate gradient 사용 (NPX 기본값) |
| `bias` | `False` | NPX 양자화(QAT) 호환. RTL 변환 시 bias 미지원 |
| `reset_mechanism` | `subtract` | 스파이크 발화 후 threshold만큼 차감 (NPX cfg 기본값) |

### 3-3. Forward Pass 패턴

NPX Trainer에서는 모델의 `forward()`가 **단일 timestep**만 처리하고, 외부 `forward_pass()` 함수에서 timestep 루프를 관리합니다:

```python
# 1. 모든 LIF 뉴런의 hidden state 초기화
utils.reset(net)

# 2. T=50 스텝 동안 동일한 입력을 반복 주입 (Constant Current)
for step in range(num_steps):
    spk_out = net(data)       # 단일 timestep forward
    spk_rec.append(spk_out)   # 출력 스파이크 기록

# 3. 모든 timestep의 스파이크를 스택
spk_rec = torch.stack(spk_rec)  # [T, batch, 6]
```

이 방식은 NPX Trainer의 `MATRIX3D` 데이터 포맷 분기와 완전히 동일합니다.

---

## 4. 학습 설정

| 항목 | 값 |
|------|-----|
| **Loss 함수** | `SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)` |
| **Optimizer** | `Adam` (기본 lr=0.001, betas=(0.9, 0.999)) |
| **Batch Size** | 64 |
| **Epochs** | 20 |
| **Time Steps** | 50 |
| **Checkpoint 형식** | `{'npx_module': state_dict, 'optimizer': optimizer_state_dict}` |

### Loss 함수 설명
`mse_count_loss`는 각 클래스 뉴런의 총 스파이크 카운트를 기반으로 MSE Loss를 계산합니다:
- **정답 클래스** 뉴런: 전체 timestep의 **80%** 만큼 스파이크하도록 유도
- **오답 클래스** 뉴런: 전체 timestep의 **20%** 만큼만 스파이크하도록 억제

---

## 5. 실행 방법 가이드 (Execution Workflow)

모든 작업은 터미널을 통해 가상 환경을 활성화 한 뒤 프로젝트의 최상위(`spike_nn/`) 디렉토리 위치에서 실행합니다.

### Step 1: 데이터 시각화 및 검증 (Data EDA)
센서의 피처들이 10개의 Batch (36개월의 시간) 단위로 구별되어 있습니다. 클래스(가스 종류) 간의 분산과 시간에 따른 센서 Drift 변화의 현상을 그래픽으로 먼저 확인할 수 있습니다.

```bash
python utils/drift_gas_sensor/visualize_drift_gas.py
```
*완료되면 `analyze/drift_gas_sensor/` 내부에 PCA 차원 축소 시각화 자료들이 생성됩니다.*


### Step 2: 모델 학습 진행 (Training SNN)
`train/train_drift_gas.py` 를 실행하여 모델 학습을 시작합니다. 

```bash
python train/train_drift_gas.py
```
**[학습 시나리오]**
- **Train:** `batch1.dat` (가장 초기에 센서가 신선할 때 측정된 데이터 그룹)
- **Test:** `batch2.dat` (시간이 다소 흘러 미세한 드리프트가 진행되기 시작한 데이터 그룹)

*학습이 완료되면 `model_weights/drift_gas_sensor/best_drift_gas_snn.pth` 에 최고 성능(Test Acc 기준) 모델의 가중치가 저장됩니다.*

**[학습 결과 예시]**
- Train Accuracy: ~97.5%
- Test Accuracy (Batch 2): ~89.9%


### Step 3: Drift 예측 시각화 (Drift Prediction)
학습된 모델로 Batch 1~10 전체에 대한 정확도를 측정하고, 시간에 따른 정확도 하락을 시각화합니다.

```bash
python utils/drift_gas_sensor/visualize_drift_prediction.py
```

*완료되면 `analyze/drift_gas_sensor/model_drift_accuracy_profile.png` 에 Batch별 정확도 그래프가 생성됩니다.*

**[결과 예시]**
| Batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|------|------|------|------|------|------|------|------|------|------|
| Acc % | 96.4 | 89.9 | 76.4 | 54.0 | 35.5 | 26.8 | 26.1 | 14.0 | 45.7 | 46.9 |

Batch 1(학습 데이터)에서 96.4%이던 정확도가 시간이 흐를수록 센서 드리프트로 인해 Batch 8에서 14.0%까지 하락합니다.

---

## 6. NPX Trainer 통합 방법

본 학습 코드는 NPX Trainer와 동일한 학습 방식을 사용하므로, 추후 통합 시 다음 두 가지만 추가하면 됩니다:

### 6-1. cfg 파일 생성
`drift_gas_app.cfg` 파일을 생성하여 npx_trainer의 `app/` 디렉토리에 배치합니다:

```ini
[preprocess]
input=drift_gas_dataset
timesteps=50
step_generation=direct

[train]
epoch=20
kfold=5
repeat=1

[global]
input_size=128,1
input_channels=1
output_classes=6
neuron_type=q8ssf

[Linear]
in_features=128
out_features=256

[Leaky]
beta=0.95
reset_mechanism=subtract
learn_threshold=False

[Linear]
in_features=256
out_features=128

[Leaky]
beta=0.95
reset_mechanism=subtract
learn_threshold=False

[Linear]
in_features=128
out_features=6

[Leaky]
beta=0.95
reset_mechanism=subtract
learn_threshold=False
```

### 6-2. NpxDataManager 분기 추가
`npx_drift_gas_data.py` 모듈의 `DriftGasNpxDataset` 클래스를 `npx_data_manager.py`에 import하고, `NpxDataManager.__init__()`에 `drift_gas` 분기를 추가합니다. 자세한 코드는 `npx_drift_gas_data.py` 파일 하단의 주석을 참고하세요.

---

## 7. 커스텀 튜닝 및 최적화

현재 분류 학습의 셋업은 최초 모델링 구동 및 Drift의 영향을 파악하기 위해 단순하게 구성되어 있습니다.
모델 성능(과적합 등)을 개선하고 싶다면 `train_drift_gas.py` 내의 설정에서 다음을 조절해 볼 수 있습니다.

* **시간 변수 (num_steps):** 50으로 되어 있는 `num_steps` 을 100 수준으로 늘리면 뉴런 내부에서 더 정밀한 스파이크 카운트가 이루어져 정확도가 소폭 상승할 수 있습니다. 
* **훈련 데이터 확장:** `train_batches=[1]` 을 `train_batches=[1, 2, 3, 4, 5]` 형태 등으로 늘리고, `test_batches=[6, 7]` 로 먼 미래의 Drift 데이터를 예측하도록 더욱 도전적인 시나리오를 구성할 수 있습니다.
* **Loss 함수 correct/incorrect rate:** `correct_rate=0.8, incorrect_rate=0.2` 값을 조절하여 스파이크 분포를 변경할 수 있습니다.
* **Beta (막전위 감쇄율):** `beta=0.95`를 조절하면 뉴런의 기억 지속 시간이 변합니다. 높을수록 이전 입력의 영향이 오래 지속됩니다.
