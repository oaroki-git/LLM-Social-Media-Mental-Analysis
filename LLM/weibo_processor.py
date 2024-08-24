import os
import mysql.connector
from mysql.connector import errorcode
import json
import time
from glm import GLM 

class WeiboProcessor:
    def __init__(self, db_config, glm_model, batch_size=2):
        self.db_config = db_config
        self.glm_model = glm_model
        self.batch_size = batch_size
        self.last_processed_id = self.load_last_processed_id()
        self.max_retries = 3
        self.connect_db()
        #self.create_output_table()

    def connect_db(self):
        try:
            self.cnx = mysql.connector.connect(**self.db_config)
            self.cursor = self.cnx.cursor()
            print("Connected to database.")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    def load_last_processed_id(self):
        # Load the last processed ID from a file (or set it to 0 if not found)
        try:
            with open('last_processed_id.txt', 'r') as f:
                return int(f.read().strip())
        except FileNotFoundError:
            return 0

    def save_last_processed_id(self, last_id):
        # Save the last processed ID to a file
        with open('last_processed_id.txt', 'w') as f:
            f.write(str(last_id))

    def create_output_table(self):
        create_table_query = """
            CREATE TABLE IF NOT EXISTS analysis_results (
            id VARCHAR(20) NOT NULL,
            ip VARCHAR(100),
            微博正文 TEXT,
            话题 VARCHAR(200),
            转发数 INT,
            评论数 INT,
            点赞数 INT,
            发布时间 DATETIME,
            measure VARCHAR(100),
            value INT,
            消极程度 INT,
            PRIMARY KEY (id, measure)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"""
        
        self.cursor.execute(create_table_query)
        self.cnx.commit()

    def process_batch(self):
        select_query = """
            SELECT id, text, topics, screen_name, created_at, reposts_count, comments_count, attitudes_count, ip
            FROM weibo
            WHERE id > %s 
            ORDER BY id ASC
            LIMIT %s
        """
        self.cursor.execute(select_query, (self.last_processed_id, self.batch_size))
        rows = self.cursor.fetchall()

        if not rows:
            print("No more data to process.")
            return
        
        output_data = []
        for row in rows:
            id, text, topics, screen_name, created_at, reposts_count, comments_count, attitudes_count, ip = row

            input_data = {
                'title': topics or "",
                'comment': text
            }
            print(input_data)

            result = self.glm_model.run(json.dumps(input_data))
            print(result)
            for measure, value in result.items():
                if measure != '消极程度':
                    entry = {
                        'id': id,
                        'ip': ip,
                        '微博正文': text,
                        '话题': topics,
                        '转发数': reposts_count,
                        '评论数': comments_count,
                        '点赞数': attitudes_count,
                        '发布时间': created_at,
                        'measure': measure,
                        'value': value,
                        '消极程度': result.get('消极程度', 0)
                    }
                    output_data.append(entry)
        # print(output_data)
        insert_query = """
            INSERT INTO analysis_results 
            (id, ip, 微博正文, 话题, 转发数, 评论数, 点赞数, 发布时间, measure, value, 消极程度)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        retries = 0
        while retries < self.max_retries:
            try:
                for entry in output_data:
                    self.cursor.execute(insert_query, (
                        entry['id'], 
                        entry['ip'], 
                        entry['微博正文'], 
                        entry['话题'], 
                        entry['转发数'], 
                        entry['评论数'], 
                        entry['点赞数'], 
                        entry['发布时间'], 
                        entry['measure'], 
                        entry['value'], 
                        entry['消极程度']
                    ))
                self.cnx.commit()
                break  # 成功时退出循环
            except mysql.connector.Error as err:
                print(f"Error during insert operation: {err}")
                if err.errno == 1205:  # 锁等待超时错误码
                    retries += 1
                    print(f"Retrying due to lock wait timeout... attempt {retries}/{self.max_retries}")
                    time.sleep(2)  # 等待一段时间再重试
                else:
                    # 打印错误并终止处理
                    print(f"Database error: {err}")
                    break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

        self.last_processed_id = rows[-1][0]
        self.save_last_processed_id(self.last_processed_id)

    def run(self, number = 10):
        cnt = 0
        while number > cnt:
            self.process_batch()
            time.sleep(2)
            cnt+=1

    def close(self):
        self.cursor.close()
        self.db_connection.close()
