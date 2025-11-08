# Talaria-AI

## 환경 세팅
1. python -m venv .venv ( 가상환경 생성  1회만 )
2. .venv\Scripts\activate ( 가상환경 활성화 )
3. pip install -r requirements.txt ( 필수 라이브러리 설치 1회만 )
4. uvicorn app.main:app --reload --port 8000 ( 파이썬 서버 실행 )

### 라이브러리 업데이트 후
venv 사용시
```
pip freeze > requirements.txt
```

conda 사용시
```
conda env export > environment.yml
```

## livekit agent 실행
개발모드로 실행
```
python src/talaria/livekit_agent/agent.py dev
```

임시파일 실행
```
python src/talaria/livekit_agent/temp.py dev 
```

운영모드로 실행
```
python src/talaria/livekit_agent/agent.py start
```

모델 다운로드 (실행 전 최초 1회)
```
python src/talaria/livekit_agent/agent.py download-files
```