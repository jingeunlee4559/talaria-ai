# 1단계: 웹 크롤링

import requests
from bs4 import BeautifulSoup
import os, csv, time

# Gartner Glossary의 기본 URL을 정의합니다.
LOCAL_FILE_PATH = "./data_pipeline/input/gartner_glossary_raw.html"
BASE_URL = "https://www.gartner.com"


def get_all_term_links_from_local():
    """
    각 알파벳 별 html을 모은 gartner_glossary_raw.html에서
    용어별 상세 링크를 추출합니다.
    """

    if not os.path.exists(LOCAL_FILE_PATH):
        print(f"error : 로컬 파일 '{LOCAL_FILE_PATH}'를 찾을 수 없습니다. 파일을 생성해주세요.")
        return []
    
    with open(LOCAL_FILE_PATH, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    term_links = []

    term_list_containers = soup.find_all('div', class_='keysearch-result-container row')

    print(f"발견된 용어 섹션 컨테이너 수: {len(term_list_containers)}")

    if term_list_containers:
        for container in term_list_containers:
            # 각 컨테이너 내의 모든 링크 태그(<a>)를 찾습니다.
            links = container.find_all('a', href=True)
            for link in links:
                # 링크는 이미 절대 경로이므로 바로 추가합니다.
                term_links.append(link['href'])
                
        print(f"총 발견된 용어 링크 수: {len(term_links)}")
    else:
        print("경고: 통합된 파일에서 용어 목록 컨테이너를 찾지 못했습니다. (클래스 이름이나 HTML 복사 오류)")

    return term_links


def get_term_definition(term_url):
    """
    각 용어 상세 페이지에서 단어와 정의를 추출합니다.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        # 요청이 Glossary 페이지에서 넘어온 것처럼 위장합니다.
        'Referer': 'https://www.gartner.com/en/information-technology/glossary',
        # 한국어 및 미국 영어 사용자로 위장합니다.
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(term_url, timeout=10, headers=headers)
        response.raise_for_status()
        time.sleep(2)

        soup = BeautifulSoup(response.text, 'html.parser')
        
        term_tag = soup.find('h1') 
        term = term_tag.get_text(strip=True) if term_tag else "N/A"
        
        definition_tag = soup.find('div', class_='col-md-12 col-sm-12 rte') 
        definition = definition_tag.get_text(strip=True) if definition_tag else "정의를 찾을 수 없음"
        
        return term, definition
    
    except Exception as e:
        print(f"오류 발생 ({term_url}): {e}")
        return "N/A", "오류로 인해 추출 실패"
    
    


def main():
    """
    전체 크롤링 프로세스를 실행하고 결과를 저장합니다.
    """
    all_terms_data = []
    
    # 1. 모든 용어 url 찾기
    term_urls = get_all_term_links_from_local()
    term_urls = term_urls[:5]

    print(f"총 {len(term_urls)}개의 용어 링크 발견.")
    print(term_urls)

    # 2. 용어별 단어와 뜻 찾기
    for term_url in term_urls:
        term, definition = get_term_definition(term_url)
        
        # 크롤링 진행 상황을 출력합니다.
        print(f"  - 추출: {term} -> {definition[:50]}...")
        
        all_terms_data.append({
            '단어': term,
            '정의': definition,
            '출처 URL': term_url
        })
        
            
    # 3. 결과 출력
    print("\n--- 크롤링 완료 ---")
    print(f"총 {len(all_terms_data)}개의 용어 데이터 추출.")

    # # 4. 파일에 저장
    # OUTPUT_DIR = './data_pipeline/output'
    # OUTPUT_FILENAME = 'gartner_glossary.csv'
    # FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
    #     print(f"폴더 '{OUTPUT_DIR}'를 생성했습니다.")

    # with open(FILE_PATH, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f, fieldnames=['단어', '정의', '출처 URL'])
    #     writer.writeheader()
    #     writer.writerows(all_terms_data)
    # print("데이터가 gartner_glossary.csv 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()