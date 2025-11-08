# %%
# src/talaria/hermes_service/rag_processor/chunker.py

import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def chunk_text(
    text: str, 
    chunk_size: int = CHUNK_SIZE, 
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    parse_document()로 추출된 긴 텍스트를
    의미 단위의 작은 텍스트 조각(Chunk) 리스트로 분할합니다.
    """
    
    if not text:
        log.warning("청킹할 텍스트가 비어있습니다.")
        return []

    # (⭐️ 핵심 로직은 동일합니다 - \n\n을 기준으로 자릅니다)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    try:
        chunks = text_splitter.split_text(text)
        
        log.info(f"문서가 {len(chunks)}개의 청크로 분할되었습니다. (Chunk Size: {chunk_size})")
        
        final_chunks = [chunk for chunk in chunks if len(chunk) > 10]
        
        return final_chunks
        
    except Exception as e:
        log.error(f"텍스트 청킹 중 오류 발생: {e}", exc_info=True)
        return []


