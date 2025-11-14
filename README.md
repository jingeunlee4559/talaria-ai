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

## 파이썬 서버 실행
```
python -m uvicorn talaria.hermes_service.main:app --reload --port 8000 --app-dir src --env-file .env

# 디버그 로그 레벨 실행
python -m uvicorn talaria.hermes_service.main:app --reload --port 8000 --app-dir src --env-file .env --log-level debug
```
### swagger
http://localhost:8000/docs#/translation/translate_doc_jobs_translate_doc_post


## livekit agent 실행
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

-----

# Talaria-AI (Hermes Project)

본 프로젝트는 LiveKit 기반의 실시간 다국어 회의를 위한 **듀얼 모드(Dual-Mode) AI 번역 에이전트** 시스템입니다.

단순한 번역기를 넘어, 회의의 성격(Business vs. Culture)에 따라 AI가 맥락을 추론하고(RAG, STT 오류 보정), 전문 용어를 보정하며, K-Culture/밈(Meme) 뉘앙스에 맞는 'Vibe 번역'과 '문화 설명'을 실시간으로 제공하는 것을 목표로 합니다.

## 🚀 핵심 기능

  * **실시간 다국어 STT 및 번역:** LiveKit의 오디오 스트림을 받아 실시간 STT(Clova) 및 Papago 1차 번역을 수행합니다.
  * **듀얼 모드(Dual-Mode) 컨텍스트 번역:** JWT 토큰의 `meeting_mode`('B'/'C') 플래그에 따라 두 가지 다른 LangGraph 파이프라인을 실행합니다.
      * **👔 비즈니스 모드 (Business Mode):** RAG, 전문 용어 사전(`expert_dict.pkl`), 이전 대화 내역을 기반으로 Papago 1차 번역의 **정확도를 보정**합니다.
      * **😎 컬처 모드 (Vibe Mode):** 2-LLM 아키텍처를 사용하여 STT 오류(예: "베그" -\> "배그")를 보정하고, K-Culture 밈/슬랭(예: "국밥", "킹받네")을 Vibe 톤으로 **스타일 번역**합니다.
  * **병렬 부가 정보 (Parallel Information):**
      * **번역 (`topic="stt"`):** 보정된 번역문과 Vibe 모드용 이모지(예: 🔥)를 결합하여 발행합니다.
      * **설명 (`topic="manner"`):** 비즈니스 모드에서는 '매너 체커'의 경고를, 컬처 모드에서는 'K-Culture 밈'의 해설(예: "'고인물': A pro gamer...")을 별도 토픽으로 발행합니다.
  * **문서 RAG 파이프라인:** 업로드된 문서를 파싱, 청킹, 임베딩하여 ChromaDB에 벡터로 저장합니다. (비즈니스 모드에서 사용)

-----

## 🛠️ 시스템 아키텍처 및 핵심 흐름

본 시스템은 2개의 독립적인 서비스가 실시간으로 통신하는 구조로 이루어져 있습니다.

### 1\. `livekit_agent` (실시간 오디오/데이터 처리기)

LiveKit 회의실에 '에이전트'로 참여하여 STT 및 데이터 I/O를 담당합니다.

  * **`src/talaria/livekit_agent/agent.py`**: LiveKit 에이전트의 메인 진입점. `_publish_stt` (번역 발행) 및 `_publish_manner` (설명/매너 발행) 콜백을 정의합니다.
  * **`src/talaria/livekit_agent/handler.py`**: **[핵심 파일]** STT `Final` 이벤트가 발생할 때, `asyncio.gather`를 사용해 2개의 작업을 병렬로 실행하여 속도를 극대화합니다.
      * **Task 1 (`_run_translation_pipeline`)**:
        1.  JWT 토큰에서 `meeting_mode`('B'/'C')를 읽어옵니다.
        2.  **모든 모드**에서 Papago 1차 번역을 호출합니다. (Vibe 모드의 'Fast Path'를 위해)
        3.  `hermes_service` API (`/utterances/process`)를 호출하며 `translation_mode`와 1차 번역본을 전달합니다.
        4.  응답으로 `final_translations` (보정본), `final_emoji` (Vibe), `final_explanation` (Vibe)을 받습니다.
        5.  Vibe 모드이고 `final_explanation`이 있으면, `topic="manner"`로 '문화 설명' 페이로드를 `publish_manner_callback`을 통해 발행합니다.
        6.  번역문과 이모지를 결합하여 `topic="stt"`로 `publish_callback`을 통해 발행합니다.
      * **Task 2 (`_run_checker_pipeline`)**:
        1.  `meeting_mode`를 확인합니다.
        2.  `mode == "business"`일 때만, `mchecker_business` (기존 매너 체커)를 실행하여 `topic="manner"`로 '매너 경고'를 발행합니다.
        3.  `mode == "vibe"`일 때는 즉시 `return`하여 Task 1이 '문화 설명'을 발행하도록 둡니다.
  * **`src/talaria/livekit_agent/manner_checker.py`**: 비즈니스 모드용 매너 체커. 시작 시 백엔드 DB(매너 DB)에서 규칙을 캐시하고, STT 원문에 대해 `contains` 검사를 수행합니다.
  * **`src/talaria/livekit_agent/integrations.py`**: `translate_to_multiple_languages` (Papago API 병렬 호출) 로직이 구현되어 있습니다.

### 2\. `hermes_service` (FastAPI + LangGraph AI 브레인)

`livekit_agent`의 요청을 받아 실제 AI 추론을 수행하는 FastAPI 서버입니다.

  * **`src/talaria/hermes_service/main.py`**: FastAPI 서버 실행 파일. `/utterances`, `/rag` 등 라우터를 포함합니다.
  * **`src/talaria/hermes_service/services/utterance_service.py`**: Agent로부터 `UtteranceProcessIn` 스키마를 받아 `TranslateState` 객체를 생성하고 LangGraph 앱을 호출(`invoke`)합니다.
  * **`src/talaria/hermes_service/state.py`**: LangGraph의 "메모리" 역할을 하는 `TranslateState` TypedDict가 정의되어 있습니다. (Business/Vibe 모드 플래그 및 결과 필드 포함)

#### 🧠 `src/talaria/hermes_service/graph.py` (핵심 로직)

현재 "Vibe 모드 하이브리드" (9-Node) 아키텍처로 구성되어 있습니다.

1.  **[진입점] `mode_router` (라우터 1)**

      * `translation_mode`("business" or "vibe")에 따라 2개 파이프라인으로 분기합니다.

2.  **[Flow A] 👔 비즈니스 모드**

      * **`analyze_business_context`** (노드): `node_check_conditions.py`. "인사말 필터", Aho-Corasick(`expert_dict.pkl`), RAG 벡터 검색을 수행하여 `should_correct_business` 플래그를 설정합니다.
      * **`business_correction_router`** (라우터 2): 위 플래그 값에 따라 분기합니다.
      * **`correction_agent_business`** (Slow Path, LLM): `node_context_translate.py`. 컨텍스트(RAG, 용어, 대화)를 힌트로 Papago 1차 번역을 '보정'합니다.
      * **`fallback_agent`** (Fast Path, 규칙): Papago 1차 번역을 `final_answer`로 복사합니다.

3.  **[Flow B] 😎 컬처(Vibe) 모드**

      * **`vibe_extract_agent`** (Slow Path, LLM 1): `node_vibe_extract.py`. STT 원문("킹밧네")을 받아 STT 오류 보정("킹받네") 및 감정 태그("anger")를 추출합니다.
      * **`context_enrich_agent`** (Slow Path, 규칙): `node_vibe_enrich.py`. `MannerChecker`의 `contains` 로직을 재활용하여 **K-Culture DB** (예: `tl_manner_rules`)를 **`contains` 검색**합니다. LLM 2(번역)를 위한 `vibe_context`(힌트)와 UI용 `final_explanation`을 생성합니다.
      * **`vibe_translation_router`** (라우터 3): `vibe_context`에 힌트가 있는지 여부에 따라 분기합니다.
      * **`correction_agent_vibe`** (Slow Path, LLM 2): `node_vibe_translate.py`. **K-Meme 힌트**를 받아 "국밥"을 "reliable"로 정확하게 Vibe 톤 번역합니다.
      * **`fallback_agent`** (Fast Path, 규칙): K-Meme이 없는 단순 문장(예: "오늘 날씨 좋네요")의 Papago 1차 번역을 `final_answer`로 복사합니다.
      * **`emoji_agent`** (종료 전): `LLM 1`이 분석한 감정 태그를 `emotion_emoji_map.json`과 매핑하여 `final_emoji`를 생성합니다.

-----

## 📁 폴더 및 파일 구조

```
talaria-ai/
├── persistent_data/                 # (Git 무시) ChromaDB 벡터 저장소
│   └── chroma_db_store/
├── scripts/                         # 🗂️ 데이터 파이프라인 및 테스트 스크립트
│   ├── assets/                      # (Git 무시됨) 원본 데이터 소스 (예: gartner_glossary_raw.html)
│   ├── input/                       # 
│   │   └── term_data.csv            # [Data] IT 전문 용어 원본 CSV
│   ├── output/                      #
│   │   ├── definitions.csv          # [Data] 전처리된 '개념' (tl_dict_concept용)
│   │   └── terms.csv                # [Data] 전처리된 '용어' (tl_dict_term용)
│   ├── steps/                       # (참고용) 데이터 파이프라인 단계별 스크립트
│   ├── build_automaton.py           # [▶️ 실행] MySQL(tl_dict_term)에서 'expert_dict.pkl' 생성 스크립트
│   ├── load_to_db.ipynb             # [▶️ 실행] output/ CSV 파일을 MySQL DB로 적재
│   ├── term_preprocessing.ipynb     # [▶️ 실행] input/term_data.csv를 output/ 2개 파일로 전처리
│   ├── test_business_graph.py       # [▶️ 실행] Business 모드 LangGraph 테스트
│   └── test_vibe_graph.py           # [▶️ 실행] Vibe 모드 LangGraph 테스트
└── src/talaria/
    ├── assets/                      # 📦 AI 모델/규칙 에셋
    │   ├── expert_dict.pkl          # [Aho-Corasick] Business 모드 전문 용어 사전
    │   └── emotion_emoji_map.json   # [Rule] Vibe 모드 감정-이모지 매핑
    ├── common/                      # 🔄 공통 모듈
    │   └── integrations/papago_doc.py # Papago 문서 번역 API 클라이언트
    ├── core/                        # 
    │   └── database.py              # [핵심] MySQL DB 연결 커서(get_db_cursor) 제공
    ├── hermes_service/              # 🧠 "브레인" (FastAPI + LangGraph)
    │   ├── nodes/                   # 🧩 LangGraph의 개별 실행 노드
    │   │   ├── node_check_conditions.py   # [Biz] Aho-Corasick, RAG, 인사말 필터 (analyze_business_context)
    │   │   ├── node_context_translate.py  # [Biz] LLM 보정 (correction_agent_business)
    │   │   ├── node_vibe_extract.py       # [Vibe] LLM 1: STT 보정 + 감정 추출
    │   │   ├── node_vibe_enrich.py        # [Vibe] Rule: K-Culture DB 'Contains' 검색
    │   │   └── node_vibe_translate.py   # [Vibe] LLM 2: 힌트 기반 번역
    │   ├── rag_processor/           # 📚 RAG 파이프라인 (파싱, 청킹, 임베딩, DB 저장)
    │   ├── routers/                 # 🌐 FastAPI 라우터
    │   │   ├── utterance.py         # [핵심] /utterances/process (LangGraph 호출 API)
    │   │   ├── docs_translation.py  # Papago 문서 번역 API
    │   │   └── rag.py               # RAG 파이프라인 실행 API
    │   ├── schemas/                 # 📝 Pydantic API 스키마
    │   │   └── utterance.py         # UtteranceProcessIn (입력), UtteranceProcessOut (출력)
    │   ├── services/                # 🛠️ FastAPI 비즈니스 로직
    │   │   └── utterance_service.py # [핵심] Agent 입력을 받아 TranslateState 생성, LangGraph 실행, 결과 반환
    │   ├── graph.py                 # [⭐️ 핵심] 9-Node 듀얼 모드 LangGraph 아키텍처 정의
    │   ├── main.py                  # [▶️ 실행] FastAPI 서버 (Uvicorn)
    │   └── state.py                 # [⭐️ 핵심] LangGraph의 메모리(TranslateState) 정의
    └── livekit_agent/               # 🎙️ "에이전트" (LiveKit 워커)
        ├── handler.py               # [⭐️ 핵심] STT 스트림 처리, Task 1/2 병렬 실행
        ├── agent.py                 # [▶️ 실행] LiveKit 에이전트 진입점
        ├── integrations.py          # [핵심] Papago 실시간 번역 (asyncio.gather)
        ├── manner_checker.py        # [Biz] 비즈니스 모드 매너 체커 (DB 캐시 + Contains)
        └── utils.py                 # 언어 코드 변환 등 유틸리티
```