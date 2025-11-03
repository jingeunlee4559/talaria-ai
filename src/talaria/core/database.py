# DB 객체를 모든 파일에서 선언하지 않도록 처리하는 매니저

import mysql.connector
from contextlib import contextmanager
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# DB 접속 정보 (이 파일에서만 관리)
DB_CONFIG = {
    'host': os.environ.get('DB_URL'),
    'user': os.environ.get('DB_USERNAME'),
    'password': os.environ.get('DB_PASSWORD'),
    'port': 3307,
    'database': 'assist',
    'charset': 'utf8mb4'
}

@contextmanager
def get_db_cursor(commit=False, dictionary=False):
    """
    DB 커넥션과 커서를 관리하는 Context Manager
    
    Args:
        commit (bool): True로 설정 시, with 블록 종료 후 자동 commit
        dictionary (bool): True로 설정 시, 딕셔너리 커서 반환
    """
    mydb = None
    mycursor = None
    try:
        # 1. 연결 및 커서 생성
        mydb = mysql.connector.connect(**DB_CONFIG)
        mycursor = mydb.cursor(dictionary=dictionary)
        
        # 2. (핵심) with 블록으로 커서를 "양보(yield)"
        yield mycursor 
        
        # 3. (성공 시) with 블록이 끝나면 자동 커밋
        if commit:
            mydb.commit()

    except mysql.connector.Error as err:
        print(f"❌ MySQL 오류: {err}")
        # 4. (실패 시) 자동 롤백
        if mydb:
            mydb.rollback()
        raise # 오류를 다시 발생시켜 상위 코드가 알 수 있게 함
    
    except Exception as e:
        print(f"❌ 기타 오류: {e}")
        if mydb:
            mydb.rollback()
        raise

    finally:
        # 5. (항상) 자동 연결 종료
        if mycursor:
            mycursor.close()
        if mydb and mydb.is_connected():
            mydb.close()
            # print("ℹ️ DB 연결 종료됨") # (너무 자주 찍히니 주석 처리)