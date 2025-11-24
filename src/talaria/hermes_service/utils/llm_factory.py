# src/talaria/hermes_service/utils/llm_factory.py
import logging
from typing import Dict
from langchain_openai import ChatOpenAI

log = logging.getLogger(__name__)

class LLMFactory:
    """
    LangChain ChatOpenAI 인스턴스를 관리하는 싱글톤 팩토리입니다.
    동일한 모델 설정(model_name, temperature)에 대해 중복 생성을 방지합니다.
    """
    _instance = None
    _llm_cache: Dict[str, ChatOpenAI] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
            log.info("✅ LLMFactory 싱글톤 인스턴스 생성됨.")
        return cls._instance

    def get_llm(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
        """
        요청한 설정에 맞는 LLM 인스턴스를 반환합니다.
        이미 생성된 인스턴스가 있다면 캐시된 것을 반환합니다.
        """
        # 캐시 키 생성 (모델명과 온도를 조합)
        cache_key = f"{model_name}_{temperature}"

        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        try:
            # log.info(f"🆕 새로운 LLM 인스턴스 생성 (Model: {model_name}, Temp: {temperature})")
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_retries=2
            )
            self._llm_cache[cache_key] = llm
            return llm
        except Exception as e:
            log.critical(f"❌ LLM 초기화 실패: {e}", exc_info=True)
            raise e

# 전역 헬퍼 함수
def get_llm_factory() -> LLMFactory:
    return LLMFactory()