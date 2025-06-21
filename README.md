# MLOps Starter Kit

취업을 위한 MLOps 프로젝트 데모입니다. Docker, CI/CD, 모델 서빙 등 MLOps의 핵심 개념을 구현했습니다.

## 🚀 주요 기능

- **모델 학습 파이프라인**: Scikit-learn을 사용한 간단한 ML 모델
- **Docker 컨테이너화**: 학습과 서빙을 위한 분리된 Docker 이미지
- **REST API**: FastAPI를 사용한 모델 서빙
- **CI/CD**: GitHub Actions를 통한 자동화
- **모델 버저닝**: 학습된 모델의 버전 관리

## 📁 프로젝트 구조

```
├── src/
│   ├── train.py          # 모델 학습 스크립트
│   ├── predict.py        # FastAPI 서버
│   └── model.py          # 모델 클래스 정의
├── models/               # 학습된 모델 저장
├── data/                 # 데이터셋
├── docker/
│   ├── Dockerfile.train  # 학습용 Docker 이미지
│   └── Dockerfile.serve  # 서빙용 Docker 이미지
├── docker-compose.yml    # 로컬 개발 환경
└── .github/workflows/    # CI/CD 파이프라인
```

## 🛠️ 시작하기

### 1. 모델 학습

```bash
docker-compose run train
```

### 2. API 서버 실행

```bash
docker-compose up serve
```

### 3. 예측 테스트

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## 📊 기술 스택

- **ML Framework**: Scikit-learn
- **API Framework**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Python**: 3.9+

## 🎯 MLOps Best Practices

1. **재현성**: Docker를 통한 환경 일관성
2. **자동화**: CI/CD 파이프라인
3. **모니터링**: 로깅 및 메트릭 수집
4. **버전 관리**: 코드와 모델 버저닝
5. **API 문서화**: FastAPI 자동 문서 생성

## 📝 License

MIT License