# Talaria-AI (Hermes Project)

**Talaria-AI**는 LiveKit 기반의 실시간 다국어 화상 회의를 위한 **Dual-Mode AI 에이전트 & 지능형 회의 분석 시스템**입니다.

단순한 통번역을 넘어, 회의의 성격(Business vs. Culture)에 따라 AI가 맥락을 파악하여 실시간으로 개입하고, 회의 종료 후에는 \*\*인사이트가 담긴 구조화된 리포트(요약, 액션 아이템, 리스크 진단)\*\*를 제공합니다.

-----

## 🚀 핵심 기능 (Key Features)

### 1\. 실시간 듀얼 모드 번역 (Dual-Mode Translation)

  * **👔 비즈니스 모드 (Business Mode):**
      * IT 전문 용어 사전(`expert_dict.pkl`)과 RAG를 활용하여 STT 오인식을 보정합니다.
      * 불필요한 개입을 최소화하는 Fast-Path 로직이 적용되어 있습니다.
  * **😎 컬처 모드 (Vibe Mode):**
      * K-Culture/밈(Meme) 용어를 감지하여 뉘앙스를 살린 'Vibe 번역'을 제공합니다.
      * 문화적 배경 지식이 필요한 경우, '문화 설명(Explanation)' 카드를 다국어로 번역하여 실시간 전송합니다.

### 2\. 스마트 회의 요약 & AI 코치 (Smart Summary & AI Coach) ✨ *New*

회의 녹취록을 분석하여 단순 요약이 아닌, **실행 가능한(Actionable) 데이터**를 생성합니다. (3-Track Hybrid Architecture)

  * **회의 요약 (Track A):** 회의 전체를 관통하는 한 줄 소개 및 확정된 의사결정 사항 추출.
  * **액션 아이템 (Track B):** 담당자, 마감일, 중요도(Priority)를 분석하여 **상위 3개 핵심 과제**를 선별.
  * **AI 리스크 진단 (Track C):** 조직행동론 기반 4대 지표(**실행 명확성, 소통 균형, 갈등 해결, 시간 효율**)로 회의 건전성을 점수화하고 개선책을 코칭.

### 3\. 문서 번역 및 RAG (Doc Translation & RAG)

  * **문서 번역:** Papago API를 활용한 문서 번역 (DB 캐싱 및 중복 방지 로직 적용).
  * **RAG 파이프라인:** 업로드된 문서를 파싱, 청킹, 임베딩하여 VectorDB(ChromaDB)에 저장하고, 실시간 회의 시 문맥 정보로 활용.

-----

## 🛠️ 시스템 아키텍처 (Architecture)

### 1\. `livekit_agent` (Real-time Processor)

LiveKit 룸에 참여하여 오디오 스트림 처리 및 통번역 중계를 담당합니다.

  * **Language Detection:** `enko` (한영 혼용) 모델을 기반으로 하되, 텍스트 분석을 통해 실제 발화 언어를 동적으로 감지하여 번역기 오류를 방지합니다.
  * **Parallel Processing:** `asyncio.gather`를 통해 '번역(Translation)'과 '매너 체크(Manner Check)' 파이프라인을 병렬로 처리합니다.
  * **Broadcast Strategy:** 문화 설명 등 부가 정보는 모든 언어 버전을 포함하여 브로드캐스트하고, 클라이언트가 로케일에 맞춰 표시합니다.

### 2\. `hermes_service` (AI Brain - FastAPI + LangGraph)

복잡한 AI 추론과 데이터 처리를 담당하는 백엔드 서비스입니다.

#### 🧠 LangGraph Workflows

  * **Translation Graph (9-Node):** 실시간 문맥 보정 및 Vibe 번역 처리.
  * **Summary Graph (3-Track Hybrid) :**
      * **Track A (Summary):** 요약 및 키워드 생성 (Evaluator: 누락/길이 검사).
      * **Track B (Action Item):** 액션 아이템 추출 및 중요도 필터링 (Evaluator: 필수값 검사).
      * **Track C (Risk Analysis):** 2+2 하이브리드 구조 (Content 분석 LLM + Process 분석 Python)로 비용 효율적인 리스크 진단 수행.

#### ⚙️ Optimization & Utils

  * **LLM Factory:** `ChatOpenAI` 인스턴스를 싱글톤으로 관리하여 연결 효율성을 높이고 설정을 중앙화했습니다.
  * **Vector Store (Singleton):** ChromaDB 연결 및 임베딩 모델 로드 최적화.

-----

## 📁 폴더 및 파일 구조 (File Structure)

주요 변경 사항 및 신규 파일이 포함된 구조입니다.

```
talaria-ai/
├── scripts/                         # 데이터 파이프라인 스크립트
├── src/talaria/
│   ├── assets/                      # AI 모델/규칙 에셋 (expert_dict.pkl, emoji maps)
│   ├── common/                      # 공통 모듈 (Papago Client 등)
│   ├── core/                        # DB 연결 (database.py)
│   ├── hermes_service/              # [AI Brain] FastAPI + LangGraph
│   │   ├── nodes/                   # 🧩 LangGraph 노드
│   │   │   ├── node_check_conditions.py   # [Biz] 번역 개입 조건 검사
│   │   │   ├── node_context_translate.py  # [Biz] 문맥 보정 번역
│   │   │   ├── node_vibe_extract.py       # [Vibe] STT 보정 및 감정 추출
│   │   │   ├── node_vibe_enrich.py        # [Vibe] K-Culture DB 검색
│   │   │   ├── node_vibe_translate.py     # [Vibe] 뉘앙스 번역
│   │   │   └── node_summary.py            # 회의 요약 및 리스크 분석 노드 (Track A/B/C)
│   │   ├── rag_processor/           # RAG 파이프라인 (Manager, Parser, Embedder, VectorStore)
│   │   ├── routers/                 # FastAPI 라우터
│   │   │   ├── utterance.py         # 실시간 발화 처리
│   │   │   ├── docs_translation.py  # 문서 번역 (Streaming/Polling)
│   │   │   └── summary.py           # 회의 요약 요청 처리 (/summary/process)
│   │   ├── schemas/                 # Pydantic 스키마
│   │   │   ├── utterance.py
│   │   │   ├── docs_translation.py
│   │   │   └── summary.py           # 요약/액션아이템/리스크리포트 구조 정의
│   │   ├── services/                # 비즈니스 로직
│   │   │   ├── utterance_service.py
│   │   │   ├── docs_translation.py  # 번역 캐싱 및 DB 저장 로직 추가
│   │   │   └── summary_service.py   # 요약 그래프 실행 및 결과 포맷팅
│   │   ├── utils/
│   │   │   ├── file_utils.py
│   │   │   └── llm_factory.py       # LLM 싱글톤 팩토리
│   │   ├── graph.py                 # 실시간 번역 그래프 정의
│   │   ├── summary_graph.py         # 회의 요약 3-Track 그래프 정의
│   │   ├── main.py                  # FastAPI 진입점
│   │   └── state.py                 # LangGraph State 정의 (SummaryState 추가됨)
│   └── livekit_agent/               # [Agent] LiveKit Worker
│       ├── agent.py                 # 에이전트 진입점
│       ├── handler.py               # 언어 감지(LID) 및 브로드캐스트 로직 강화
│       ├── integrations.py          # Papago API 연동
│       ├── manner_checker.py        # 매너 체크 로직
│       └── utils.py                 # 언어 코드 변환 유틸
├── .env                             # 환경 변수 설정
├── requirements.txt                 # 의존성 목록
└── README.md                        # 프로젝트 문서
```

-----

## 💻 실행 방법 (How to Run)

### 환경 세팅
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

### 파이썬 서버 실행
```
python -m uvicorn talaria.hermes_service.main:app --reload --port 8000 --app-dir src --env-file .env

# 디버그 로그 레벨 실행
python -m uvicorn talaria.hermes_service.main:app --reload --port 8000 --app-dir src --env-file .env --log-level debug
```
#### swagger
http://localhost:8000/docs#/translation/translate_doc_jobs_translate_doc_post


### livekit agent 실행
개발모드로 실행
```
python -m src.talaria.livekit_agent.agent dev
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

### ⚠️ 주요 환경 변수 (.env)

  * `OPENAI_API_KEY`: LLM 사용을 위해 필수.
  * `DB_URL`, `DB_USERNAME`, `DB_PASSWORD`: MySQL 연결 정보.
  * `PAPAGO_CLIENT_ID`, `PAPAGO_CLIENT_SECRET`: Papago 번역 API 키.
  * `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`: LiveKit 접속 정보.
