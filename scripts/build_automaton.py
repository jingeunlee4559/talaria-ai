import ahocorasick # (pyahocorasick 라이브러리)
import pickle
import mysql.connector
from dotenv import load_dotenv
import os

# --- 1. .env 파일 및 DB 설정 로드 ---
load_dotenv()

DB_CONFIG = {
    'host': os.environ.get('DB_URL'),
    'user': os.environ.get('DB_USERNAME'),
    'password': os.environ.get('DB_PASSWORD'),
    'port': 3307,
    'database': 'assist',
    'charset': 'utf8mb4'
}

# --- 2. DB 연결 및 오토마톤 빌드 ---
# finally에서 사용하기 위해 변수를 밖에서 None으로 초기화합니다.
mydb = None
mycursor = None

try:
    # 1. DB 연결
    mydb = mysql.connector.connect(**DB_CONFIG)
    mycursor = mydb.cursor()
    print("✅ MySQL 서버 연결 성공!")

    # 2. (핵심) DB에서 키워드 데이터 가져오기
    #    모든 실제 작업은 try 블록 안에서 수행합니다.
    print("ℹ️ DB에서 키워드 로드를 시작합니다...")
    sql = "SELECT term_text, concept_id FROM tl_dict_term;"
    mycursor.execute(sql)
    results = mycursor.fetchall()
    print(f"✅ {len(results)}개의 키워드 로드 완료.")

    # 3. 아호-코라식 오토마톤 생성
    automaton = ahocorasick.Automaton()

    # 4. 오토마톤에 (키워드, concept_id) 추가
    for (term, concept_id) in results:
        automaton.add_word(term, concept_id)

    # 5. 오토마톤 빌드 (필수)
    automaton.make_automaton()
    print("✅ 오토마톤 빌드 완료.")

    # 6. 완성된 오토마톤을 파일로 저장 (직렬화)
    with open('./output/expert_dict.pkl', 'wb') as f:
        pickle.dump(automaton, f)

    print("✅ 'expert_dict.pkl' 파일 저장 완료!")

except mysql.connector.Error as err:
    # DB 연결 오류 또는 쿼리 실행 오류
    print(f"❌ MySQL 오류: {err}")
except Exception as e:
    # pickle 저장 오류 등 기타 오류
    print(f"❌ 기타 오류 발생: {e}")

finally:
    # 7. (중요) 모든 작업이 끝나면 연결을 닫습니다.
    #    연결에 실패한 경우(mycursor=None)를 대비해 if문으로 확인합니다.
    try:
        if mycursor:
            mycursor.close()
        if mydb and mydb.is_connected():
            mydb.close()
            print("✅ MySQL 연결이 성공적으로 종료되었습니다.")
        elif mydb: # 연결은 안됐지만 객체는 생성된 경우
             pass
        else: # 연결 시도조차 못한 경우
             print("ℹ️ MySQL 연결이 시작되지 않았습니다.")
             
    except mysql.connector.Error as err:
        print(f"❌ 연결 종료 중 오류: {err}")
