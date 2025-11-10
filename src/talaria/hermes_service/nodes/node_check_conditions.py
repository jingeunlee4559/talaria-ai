# 문맥기반 번역 조건 노드

import logging
import pickle
import os
import ahocorasick
from typing import List, Optional, Set, Dict, Any
from langchain_core.documents import Document

# 1. 상태 및 RAG/DB 리소스 임포트
# (state.py의 위치에 따라 상대 경로 조정이 필요할 수 있습니다)
from ..state import TranslateState, FoundTerm
from ..rag_processor.embedder import get_embedder
from ..rag_processor.vector_store import get_vector_store
from ...core.database import get_db_cursor

log = logging.getLogger(__name__)

# --- 조건부 번역을 위한 상수 정의 ---
RAG_SIMILARITY_THRESHOLD = 0.7  # L2 거리 기준 (0.7 미만일 때 관련성 높다고 판단)
SHORT_SENTENCE_WORD_COUNT = 7   # 이 단어 개수 이하일 때 "짧은 문장"으로 간주
RAG_RESULT_COUNT = 3            # RAG 검색 시 가져올 청크 개수

# --- 리소스 경로 정의 ---
# 이 파일(node_check_conditions.py)의 위치를 기준으로 expert_dict.pkl 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERT_DICT_PATH = os.path.join(
    BASE_DIR, "..", "..", "assets", "expert_dict.pkl"
) # src/talaria/assets/expert_dict.pkl


class _ConditionChecker:
    """
    (싱글톤) 조건 검사에 필요한 무거운 리소스(모델, DB연결, 오토마톤)를
    메모리에 한 번만 로드하고 관리하는 헬퍼 클래스
    """
    def __init__(self):
        log.info("조건 검사기(_ConditionChecker) 초기화 시작...")
        try:
            # 아호-코라식 오토마톤 로드
            self.automaton = self._load_automaton(EXPERT_DICT_PATH)
            # RAG 임베더 로드
            self.embedder = get_embedder()
            # RAG 벡터 저장소 로드
            self.vector_store = get_vector_store()
            log.info("✅ 조건 검사기 리소스(오토마톤, 임베더, 벡터저장소) 로드 완료.")
        except Exception as e:
            log.error(f"❌ 조건 검사기 초기화 실패: {e}", exc_info=True)
            self.automaton = None
            self.embedder = None
            self.vector_store = None

    def _load_automaton(self, pkl_path: str) -> Optional[ahocorasick.Automaton]:
        """
        아호-코라식 오토마톤 .pkl 파일을 로드합니다.
        """
        if not os.path.exists(pkl_path):
            log.error(f"전문 용어 사전 파일({pkl_path})을 찾을 수 없습니다.")
            return None
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f) #
        except Exception as e:
            log.error(f"전문 용어 사전 로드 실패: {e}", exc_info=True)
            return None

    def _get_docs_for_meeting(self, meeting_id: int) -> Optional[List[int]]:
        """
        (Java DB 연동 1)
        meeting_id를 기반으로 Java 백엔드의 'tbl_meetings_documents' 테이블을 조회하여
        이 회의에 연결된 'document_id' 목록을 가져옵니다.
        """
        try:
            # `assist` DB에 연결합니다.
            with get_db_cursor(dictionary=True) as cursor:
                # (사용자가 제공한 실제 쿼리 사용)
                sql = """
                    SELECT document_id
                    FROM tbl_meetings_documents 
                    WHERE meeting_id = %s
                """
                cursor.execute(sql, (meeting_id,))
                results = cursor.fetchall()
                
                doc_ids = [row['document_id'] for row in results if row]
                
                if doc_ids:
                    log.info(f"[Meeting: {meeting_id}] 참조 문서 ID {len(doc_ids)}개 조회 완료: {doc_ids}")
                    return doc_ids
                else:
                    log.info(f"[Meeting: {meeting_id}] 참조 문서가 없습니다.")
                    return None
                
        except Exception as e:
            log.error(f"[Meeting: {meeting_id}] 참조 문서 ID 조회 중 DB 오류: {e}", exc_info=True)
            return None

    def _get_term_meanings(self, concept_ids: Set[int]) -> List[FoundTerm]:
        """
        (Python DB 연동 2)
        아호-코라식으로 찾은 concept_id 목록을 사용해 'assist' DB에서
        실제 용어(term_text)와 한국어 의미(description_ko)를 조회합니다.
        """
        if not concept_ids:
            return []
            
        found_terms_list: List[FoundTerm] = []
        try:
            with get_db_cursor(dictionary=True) as cursor:
                id_list_str = ", ".join(map(str, concept_ids))
                
                # tl_dict_term: 용어 텍스트 (e.g., "CR")
                # tl_dict_concept: 용어 의미 (e.g., "변경 요청서")
                sql = f"""
                    SELECT 
                        t.term_text, 
                        c.description_ko
                    FROM 
                        tl_dict_term t
                    JOIN 
                        tl_dict_concept c ON t.concept_id = c.concept_id
                    WHERE 
                        t.concept_id IN ({id_list_str})
                    GROUP BY 
                        t.concept_id, t.term_text, c.description_ko
                """
                
                cursor.execute(sql)
                results = cursor.fetchall()
                
                for row in results:
                    found_terms_list.append(
                        FoundTerm(term=row['term_text'], meaning=row['description_ko'])
                    )
            
            log.info(f"전문 용어 {len(found_terms_list)}개 의미 조회 완료.")
            return found_terms_list

        except Exception as e:
            log.error(f"전문 용어 의미 조회 중 DB 오류: {e}", exc_info=True)
            return [] # 오류 시 빈 리스트 반환

    def run(self, state: TranslateState) -> TranslateState:
        """
        LangGraph 노드에서 호출할 메인 실행 함수.
        3가지 조건을 검사하여 'should_context_translate' 플래그를 설정합니다.
        """
        if not self.automaton or not self.embedder or not self.vector_store:
            log.error("필수 리소스(오토마톤/임베더/DB)가 로드되지 않아 조건 검사를 중단합니다.")
            state['should_context_translate'] = False
            return state

        stt_text = state['stt_text']
        meeting_id = state['meeting_id']
        
        # 1. 상태값 초기화
        should_translate = False
        found_terms_list: List[FoundTerm] = []
        context_docs: List[Document] = []
        
        # --- (준비) Java DB에서 이 회의의 문서 ID 목록 가져오기 ---
        docs_id = self._get_docs_for_meeting(meeting_id)

        # --- [조건 1: 전문 용어 검사] (가벼운 검사) ---
        try:
            found_concept_ids: Set[int] = set()
            for end_index, concept_id in self.automaton.iter(stt_text):
                found_concept_ids.add(concept_id)
            
            if found_concept_ids:
                log.info(f"조건 1(전문 용어) 만족: {found_concept_ids}")
                should_translate = True # ⭐️ LLM 호출 트리거
                # DB에서 용어와 뜻을 조회하여 state에 저장
                found_terms_list = self._get_term_meanings(found_concept_ids)
                
        except Exception as e:
            log.warning(f"아호-코라식 검사 중 오류: {e}", exc_info=True)

        # --- [조건 3: 짧은 문장 검사] (가벼운 검사) ---
        try:
            word_count = len(stt_text.split())
            if word_count <= SHORT_SENTENCE_WORD_COUNT:
                log.info(f"조건 3(짧은 문장) 만족: (단어 수: {word_count})")
                should_translate = True # ⭐️ LLM 호출 트리거
        except Exception as e:
            log.warning(f"문장 길이 검사 중 오류: {e}", exc_info=True)

        # --- [조건 2: RAG 유사도 검사] (무거운 검사 - 재료 수집 겸용) ---
        rag_condition_met = False # RAG 조건이 만족되었는지 별도 플래그
        
        if docs_id: # 회의에 등록된 문서가 있을 경우에만 RAG 수행
            try:
                # 1. 발화 STT를 벡터로 변환
                query_vector = self.embedder.embed_chunks([stt_text])[0]
                
                # 2. VectorDB에서 이 회의 문서들(docs_id) 내에서만 검색
                rag_results = self.vector_store.search_chunks(
                    query_vector=query_vector,
                    document_ids=[str(doc_id) for doc_id in docs_id], # ChromaDB는 문자열 ID를 기대할 수 있음
                    n_results=RAG_RESULT_COUNT
                )

                # 3. (⭐️ 하이브리드 "약한 신호" 로직)
                if rag_results:
                    # 3a. Top-1 청크는 '약한 신호'일 수 있으므로 (유사도와 상관없이) 일단 추가
                    top_1_result = rag_results[0]
                    top_1_distance = top_1_result.get('distance', 999)
                    context_docs.append(Document(
                        page_content=top_1_result.get('text', ''),
                        metadata={'distance': top_1_distance, 'doc_id': top_1_result.get('document_id', 'unknown')}
                    ))

                    # 3b. Top-1이 임계값을 통과했는지 검사
                    if top_1_distance < RAG_SIMILARITY_THRESHOLD:
                        rag_condition_met = True
                    
                    # 3c. Top-2 부터는 '노이즈' 방지를 위해 임계값을 통과한 것만 추가
                    for result in rag_results[1:]: # 1번 인덱스부터(Top-2) 순회
                        distance = result.get('distance', 999)
                        if distance < RAG_SIMILARITY_THRESHOLD:
                            context_docs.append(Document(
                                page_content=result.get('text', ''),
                                metadata={'distance': distance, 'doc_id': result.get('document_id', 'unknown')}
                            ))
                
                # 4. RAG 조건이 만족되었는지 최종 확인
                if rag_condition_met:
                    log.info(f"조건 2(RAG 유사도) 만족: (Top-1 거리: {top_1_distance:.4f})")
                    should_translate = True # ⭐️ LLM 호출 트리거
                    
            except Exception as e:
                log.error(f"RAG 유사도 검사 중 오류: {e}", exc_info=True)
                # context_docs는 비어있을 것임 (정상)
        else:
            log.info("회의에 등록된 RAG 문서가 없어 조건 2(RAG)를 건너뜁니다.")

        # --- 최종 상태 업데이트 ---
        state['docs_id'] = docs_id
        state['context'] = context_docs
        state['found_terms'] = found_terms_list
        state['should_context_translate'] = should_translate # ⭐️ 3가지 조건 중 하나라도 만족하면 True
        
        log.info(f"조건 검사 완료: LLM 보정 필요 여부 = {should_translate}")

        return state

# --- LangGraph가 사용할 수 있도록 싱글톤 인스턴스 생성 ---
try:
    _checker_instance = _ConditionChecker()
except Exception as e:
    log.critical(f"❌ _ConditionChecker 인스턴스 생성 실패. 시스템을 시작할 수 없습니다: {e}", exc_info=True)
    _checker_instance = None # 실패 시 None으로 설정

def check_conditions_node(state: TranslateState) -> TranslateState:
    """
    LangGraph의 'check_conditions' 노드입니다.
    
    3가지 조건(전문용어, RAG유사도, 문장길이)을 검사하고
    RAG 검색을 수행하여 state를 업데이트합니다.
    """
    if not _checker_instance:
        log.error("조건 검사기가 초기화되지 않았습니다. 기본 번역으로 폴백합니다.")
        state['should_context_translate'] = False
        return state
        
    try:
        # (주의) LangGraph는 상태 객체를 '복사'해서 전달할 수 있습니다.
        # 명시적으로 반환값을 받아 state를 업데이트해야 합니다.
        updated_state = _checker_instance.run(state)
        return updated_state
    except Exception as e:
        log.error(f"check_conditions_node 실행 중 심각한 오류 발생: {e}", exc_info=True)
        # 노드 실행 중 오류가 발생하면, 안전하게 LLM 번역을 건너뛰도록 설정
        state['should_context_translate'] = False
        return state