#4단계: DB 적재
from dotenv import load_dotenv
import os
import mysql.connector

load_dotenv()

DB_CONFIG = {
    'host': os.environ.get('DB_URL'),       # DB 서버 주소 (보통 로컬이면 'localhost')
    'user': os.environ.get('DB_USERNAME'), # MySQL 사용자 이름 (예: root)
    'password': os.environ.get('DB_PASSWORD'), # MySQL 사용자 비밀번호
    'port': 3307,  # Port 번호
    'database': 'assist',  # 데이터를 넣을 데이터베이스 이름
    'charset': 'utf8mb4'       # 한글 및 다양한 문자를 위한 인코딩 설정
}

try:
    # 1. DB 연결
    mydb = mysql.connector.connect(**DB_CONFIG)
    
    # 2. 커서(Cursor) 생성: SQL 명령을 실행할 객체
    mycursor = mydb.cursor()
    
    print("✅ MySQL 서버 연결 성공!")
    
    # 여기서부터 SQL 명령을 실행합니다 (3단계 참고).

    # ---------------------------------------------------------------
    # 3. 데이터 삽입 (DML: Data Manipulation Language)
    # ---------------------------------------------------------------
    
    # [A] definitions 데이터 삽입 (Primary Key 테이블 먼저)
    # df_definition의 컬럼 순서에 맞춰 데이터를 준비합니다.
    def_data = [tuple(row) for row in df_definition[['def_id', 'English Meaning', 'Korean Meaning']].values]
    sql_insert_def = "INSERT INTO definitions (def_id, English_Meaning, Korean_Meaning) VALUES (%s, %s, %s)"
    
    mycursor.executemany(sql_insert_def, def_data) 
    print(f"✅ definitions 테이블에 {len(def_data)}개 행 삽입 완료.")

    # [B] terms 데이터 삽입 (Foreign Key 테이블 나중)
    # df_final_term의 컬럼 순서에 맞춰 데이터를 준비합니다. (term_id는 AUTO_INCREMENT이므로 제외)
    term_data = [tuple(row) for row in df_final_term[['def_id', 'Language', 'Word']].values]
    sql_insert_term = "INSERT INTO terms (def_id, Language, Word) VALUES (%s, %s, %s)"
    
    mycursor.executemany(sql_insert_term, term_data)
    print(f"✅ terms 테이블에 {len(term_data)}개 행 삽입 완료.")

    # ---------------------------------------------------------------
    # 4. 최종 반영 및 연결 종료
    # ---------------------------------------------------------------
    mydb.commit() # 모든 변경 사항을 DB에 최종 반영
    print("✨ DB 적재 및 최종 커밋 완료.")
    
except mysql.connector.Error as err:
    print(f"❌ MySQL 연결 오류: {err}")
    # 연결 실패 시 자세한 오류 메시지를 출력합니다.

finally:
    # 작업이 끝나면 연결을 닫습니다. (실제 데이터 삽입 후 수행)
    if 'mydb' in locals() and mydb.is_connected():
        mydb.close()
        pass
