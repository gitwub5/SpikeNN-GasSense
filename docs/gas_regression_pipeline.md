# Gas Sensor Regression with Spiking Neural Networks (SNN)

이 문서는 UCI "Gas Sensor Array under Dynamic Gas Mixtures" 데이터셋을 바탕으로, **연속적인 가스 농도(ppm)**를 SNN (Spiking Neural Network)을 통해 예측(Regression)하는 파이프라인의 전체 실행 방법과 학습 과정을 설명합니다.

---

## 1. 개요 (Overview)

본 파이프라인의 핵심 구조는 다음과 같습니다.
1. **입력:** 16개의 금속 산화물 화학 센서 시계열 데이터
2. **출력:** 윈도우 시퀀스의 마지막 시점에서의 `[가스 1 농도, Ethylene 농도]` (단위: ppm)
3. **모델 구조:** `Linear -> LIF -> Linear -> LIF` 구조의 SpikingNet (Leaky Integrate-and-Fire 모델)

> **왜 SNN 모델 출력으로 농도값 연속체(Continuous Value)를 어떻게 예측하나요?**
> SNN 출력층의 **마지막 Time Step의 전압(Membrane Potential, `mem_rec[-1]`)**을 스파이크 발화 여부에 관계없이 바로 농도 추정 값으로 사용하며, 오차 역전파는 순수 `MSELoss`를 사용하여 진행합니다.

---

## 2. 파이프라인 파일 및 구조

```text
spike_nn/
├── data/gas_sensor/            <-- 변환된 PyTorch Tensor (.pt) 저장소
├── data_raw/gas_sensor/        <-- 원본 텍스트 데이터 (.txt 파일들)
├── dataset/
│   └── gas_dataset.py          <-- PyTorch DataLoader용 데이터셋 코드
├── model/
│   └── spiking_net.py          <-- SNN 모델 레이어 소스
├── train/
│   └── train_gas.py            <-- Regression 목적의 모델 메인 학습 모듈
├── utils/
│   ├── preprocess_gas_data.py  <-- 텍스트 ➔ Tensor 윈도우 슬라이싱 및 정규화
│   ├── verify_net.py           <-- 정규화 데이터의 Shape 검증
│   └── visualize_regression.py <-- 학습 완료 모델의 Inference 및 차트 시각화
└── analyze/                    <-- 예측 결과 그래프 등 분석 이미지 저장소
```

---

## 3. 실행 방법 가이드 (Execution Workflow)

모든 작업은 터미널을 통해 프로젝트의 루트(`spike_nn/`) 디렉토리 위치에서 순서대로 진행합니다.

### Step 1: 원본 데이터 전처리 (Preprocessing)
수백만 줄의 txt 파일에서 500 timestep 크기의 슬라이딩 윈도우 단위로 데이터를 자르고, StandardScaler를 적용하여 `.pt` 텐서 포맷으로 압축합니다.

```bash
# 가상환경(venv)을 활성화한 상태에서 실행 (시간이 수 분 정도 소요됨)
python utils/preprocess_gas_data.py
```
*완료되면 `data/gas_sensor/` 내부에 `ethylene_CO.pt` 와 `ethylene_methane.pt` 파일이 생성됩니다.*

### Step 2: 전처리 데이터 검증 (Verification)
생성된 데이터의 차원인 `(N, 500, 16)` 입력과 `(N, 2)` 타겟 형태가 올바르게 저장되었는지 확인합니다.

```bash
python utils/verify_net.py
```

### Step 3: 모델 학습 진행 (Training)
학습 파라미터는 `train/train_gas.py` 파일 상단의 `CONFIG` 딕셔너리에서 수정할 수 있습니다.
- Mac의 경우 GPU 연산 가속인 **MPS**를 자동 지원하며, Nvidia의 경우 **CUDA**를 감지하여 훈련합니다.

```bash
python train/train_gas.py
```
*학습이 완료되면 `model_weights/snn_gas_regression.pth` 에 모델의 가중치가 저장됩니다.*

> 💡 **Tip:** 학습 시 500 타임스텝의 BPTT(역전파)가 진행되므로 에폭(Epoch)당 시간이 다소 길 수 있습니다. 코드에서 1차적인 연산 호환성이나 전체 파이프라인의 에러가 나지 않는 지 가장 빠르게 검증하고 싶을 때는 `CONFIG` 에서 `batch_size = 256`, `num_epochs = 1` 로 극단적으로 조절하면 1분 이내에 파이프라인 끝단까지 성공하는지 (Dry Run) 테스트해볼 수 있습니다. 실제 튜닝용 학습 시에는 본래 권장값(`batch_size=32`, `num_epochs=10~20`)을 사용하세요.

### Step 4: 시각화 및 추론 결과 (Inference & Visualization)
학습에 쓰이지 않은 Test Dataset (전체의 20%) 데이터를 활용하여 `model_weights/` 에 저장된 예측 모델과의 결괏값을 비교하는 그래프를 생성합니다.

```bash
python utils/visualize_regression.py
```
*완료되면 `analyze/regression_inference_plot.png` 에 그래프가 저장됩니다.* 
파란색/초록색 선이 **정답(True target)**이며 적색/주황 점선이 SNN이 추론한 센서 데이터의 **실제 가스 농도(Predicted)** 입니다.

---

## 4. 커스텀 튜닝 및 최적화
현재 타겟은 `window_size` 로 100Hz 모델링 5초의 데이터를 사용하고 있습니다. 윈도우 사이즈를 늘리거나 줄이고 싶으시면 `preprocess_gas_data.py`의 `window_size` 설정과 이 데이터들을 받아 읽는 `train_gas.py` 내 시간축 설정(`time_steps`) 값을 서로 일치시켜 재생성해주시면 됩니다.
