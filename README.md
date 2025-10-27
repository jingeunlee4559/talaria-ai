# Talaria-AI

## 환경 세팅
python -m venv .venv ( 가상환경 생성  1회만 )
.venv\Scripts\activate ( 가상환경 활성화 )
pip install -r requirements.txt ( 필수 라이브러리 설치 1회만 )
uvicorn app.main:app --reload --port 8000 ( 파이썬 서버 실행 )

pip list --format=freeze > requirements.txt (라이브러리 install 후)