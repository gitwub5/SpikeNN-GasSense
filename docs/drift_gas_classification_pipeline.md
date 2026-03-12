# Gas Sensor Classification with Spiking Neural Networks (SNN)

이 문서는 UCI "Gas Sensor Array Drift Dataset" 데이터셋을 바탕으로, **센서의 반응을 통해 6가지 가스의 종류를 분류(Classification)**하는 SNN (Spiking Neural Network) 파이프라인의 전체 구조와 과정을 설명합니다. 

특히 시간이 지남에 따라 센서 반응이 미세하게 틀어지는 **Drift 현상**을 고려하여, 초기 수집 데이터로 학습하고 나중 수집 데이터로 테스트하는 시나리오를 채택하고 있습니다.

---

## 1. 개요 (Overview)

본 파이프라인의 핵심 구조는 다음과 같습니다.
1. **입력:** 16개의 연속(Time-Series) 반응에서 사전에 추출 및 통계 요약된 **128차원의 정적 피처(Static Feature) 벡터**
2. **출력:** Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene 를 분별하는 6개 클래스 확률(Spike Count)
3. **가스 인코딩 방식:** Constant Current Input (지속적인 입력 주입)을 통해 내부적으로 스파이크 발화를 유도
4. **모델 구조:** `Linear -> LIF -> Linear -> LIF` 구조의 SpikingNet (Leaky Integrate-and-Fire 모델)

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
│       └── visualize_drift_gas.py    <-- PCA 시각화 및 피처 평균 그래프 생성 스크립트
└── analyze/drift_gas_sensor/         <-- 분류 클래스 분리 및 시간에 따른 Drift 양상 그래픽 저장소
```

---

## 3. 실행 방법 가이드 (Execution Workflow)

모든 작업은 터미널을 통해 가상 환경을 활성화 한 뒤 프로젝트의 최상위(`spike_nn/`) 디렉토리 위치에서 실행합니다.

### Step 1: 데이터 시각화 및 검증 (Data EDA)
센서의 피처들이 10개의 Batch (36개월의 시간) 단위로 구별되어 있습니다. 클래스(가스 종류) 간의 분산과 시간에 따른 센서 Drift 변화의 현상을 그래픽으로 먼저 확인할 수 있습니다.

```bash
python utils/drift_gas_sensor/visualize_drift_gas.py
```
*완료되면 `analyze/drift_gas_sensor/` 내부에 `batch1_pca_classes.png` 및 `all_batches_pca_drift.png` 와 같은 PCA 차원 축소 시각화 자료들이 생성됩니다.*


### Step 2: 모델 학습 진행 (Training SNN)
`train/train_drift_gas.py` 를 실행하여 모델 학습을 시작합니다. 
분류 문제의 경우 `snntorch`의 `ce_count_loss` 함수를 통해 지정된 `T=50` 타임 스텝 동안 총 누적 스파이크 횟수로 다중 분류 Loss를 측정합니다.

```bash
python train/train_drift_gas.py
```
**[학습 시나리오]**
- **Train:** `batch1.dat` (가장 초기에 센서가 신선할 때 측정된 데이터 그룹)
- **Test:** `batch2.dat` (시간이 다소 흘러 미세한 드리프트가 진행되기 시작한 데이터 그룹)

*학습이 완료되면 `model_weights/drift_gas_sensor/best_drift_gas_snn.pth` 에 최고 성능(Test Acc 기준) 모델의 가중치가 저장됩니다. 최고 테스트 정확도는 약 85~90% 사이에 도달합니다.*

---

## 4. 커스텀 튜닝 및 최적화
현재 분류 학습의 셋업은 최초 모델링 구동 및 Drift의 영향을 파악하기 위해 단순하게 구성되어 있습니다.
모델 성능(과적합 등)을 개선하고 싶으시다면 `train_drift_gas.py` 내의 설정에서 다음 두 가지를 조절해 볼 수 있습니다.

* **시간 변수 (num_steps):** 50으로 되어 있는 `num_steps` 을 100 수준으로 늘리면 뉴런 내부에서 더 정밀한 스파이크 카운트가 이루어져 정확도가 소폭 상승할 수 있습니다. 
* **훈련 데이터 확장:** `train_batches=[1]` 을 `train_batches=[1, 2, 3, 4, 5]` 형태 등으로 늘리고, `test_batches=[6, 7]` 로 먼 미래의 Drift 데이터를 예측하도록 더욱 도전적인 시나리오를 구성할 수 있습니다. 데이터를 여러 개 묶는 로직은 이미 `dataset.py`안에 내장되어 있습니다.
