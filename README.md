# 🌇 FashionMNIST 조건부 생성(cGAN) 구현 및 개선 실험

## 📌 프로젝트 개요
FashionMNIST(10개 클래스)에서 클래스 레이블을 조건으로 이미지를 생성하는 Conditional GAN(cGAN)을 구현하고,
Baseline을 재현한 뒤 구조 및 학습 안정화 기법을 적용하여 생성 품질 변화를 비교·분석한 프로젝트입니다.

본 프로젝트는 “GAN이 돌아간다”를 넘어서,
- 조건이 실제로 반영되는지(레이블-이미지 일치)
- 학습이 안정적으로 진행되는지(D(x), D(G(z)) 및 loss 관찰)
- 생성 품질이 개선되는지(FID/IS, optional)
를 근거 기반으로 확인하는 데 목적이 있습니다.

---

## 목표
- FashionMNIST 10개 클래스에 대해 조건부 이미지 생성 모델 구현(cGAN)
- Baseline 재현 및 한계 분석
- 구조/학습 안정화 기법 적용 후 품질 변화 비교
- 정성(샘플) + 정량(FID/IS, optional) 평가로 결과 해석

---

## 데이터
- torchvision.datasets.FashionMNIST
- 28×28 흑백(1채널), 10개 클래스
- train 60,000 / test 10,000

---

## 구현 및 실험 구성

### 1) Baseline (재현/분석)
- 조건 입력: noise z + label embedding을 결합(concat)
- 손실: BCE 기반(판별자 Sigmoid 포함)
- 관찰 포인트:
  - D(x), D(G(z)), G/D loss가 어느 구간에서 정체/불안정해지는지 확인
  - 생성 이미지의 선명도/다양성 및 레이블 반영 여부를 시각적으로 점검

### 2) 개선 실험 (구조/학습 안정화)
- 모델 구조:
  - Generator/Discriminator를 Residual block 기반 구조로 변경
  - 레이블 임베딩 차원 확장(label_dim=30)
  - Discriminator에 Dropout 적용
- 학습 전략:
  - 손실: BCE → MSE(LSGAN 스타일, Sigmoid 제거)
  - Optimizer: AdamW + weight_decay
  - Scheduler: CosineAnnealingLR
  - Label smoothing 적용(smoothing=0.1)
- 평가:
  - 클래스별 샘플 그리드로 조건 반영 여부 점검
  - 체크포인트별 FID/IS 계산으로 품질 변화 비교

---

## 결과 요약

체크포인트 자동 평가 결과 일부:
- Epoch 10: FID 39.4373, IS 4.0799 ± 0.1412
- Epoch 100: FID 16.8721, IS 4.5606 ± 0.1323

해석:
- 학습이 진행되며 FID가 감소하는 경향을 확인했다.
- 단, FashionMNIST는 자연이미지 도메인과 달라 절대값보다는 실험 간 상대 비교에 의미가 있다.

---

## 🟨 결론 및 인사이트

- 구조/학습 안정화 기법(Residual, MSE, AdamW, cosine, smoothing 등)은
  정량 지표(FID) 관점에서 개선 흐름을 만드는 데 기여했다.
- 그러나 cGAN의 핵심은 “조건 일치”이므로, 레이블과 생성 이미지의 매칭이 불안정한 사례가 존재한다면 추가 개선이 필요하다.
- 다음 실험 방향:
  - 조건 주입 방식 고도화(projection, Conditional BN, FiLM 등)
  - smoothing/dropout/lr 스윕으로 최적 조합 탐색
  - 클래스별 고정 그리드 평가를 루틴화하여 조건 일치 여부를 지속 점검


