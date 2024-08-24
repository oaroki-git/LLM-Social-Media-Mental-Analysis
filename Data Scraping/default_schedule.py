import schedule
import time
import subprocess
import os
import signal
import scrape_hotsearch
from datetime import datetime, timedelta

exit = False

# 定义要在指定时间执行的函数
def scheduled_task():
    print("Scheduled task started at:", datetime.now())
    
    # 设置输出日志文件的路径
    log_file = f"logs/scrapy_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 打开日志文件
    with open(log_file, 'w') as f:
        # 启动 Scrapy 爬虫作为子进程，并将输出重定向到日志文件
        scrape_hotsearch.run(20)
        os.system('rm -rf crawls')
        process = subprocess.Popen(
            ['scrapy', 'crawl', 'search', '-s', 'JOBDIR=crawls/search'],
            stdout=f,
            stderr=subprocess.STDOUT
        )
        print(f"Scrapy spider started with PID: {process.pid}, logging to: {log_file}")
        
        # 设定爬虫运行的时长（例如30分钟）
        duration = timedelta(minutes=60)
        end_time = datetime.now() + duration
        
        # 等待爬虫运行并监控
        while datetime.now() < end_time:
            try:
                time.sleep(5)  # 每隔5秒检查一次
            except KeyboardInterrupt:
                print("spider killed due to keybaord interrupt")
                os.kill(process.pid, signal.SIGTERM)
                exit = True
                return
        
        # 停止爬虫
        os.kill(process.pid, signal.SIGTERM)
        print(f"Scrapy spider with PID {process.pid} has been terminated. Logs saved to {log_file}.")

# 安排每天的指定时间执行函数，例如每天9:00执行任务
schedule_time = "10:00"
schedule.every().day.at(schedule_time).do(scheduled_task)
schedule_time = "22:00"
schedule.every().day.at(schedule_time).do(scheduled_task)

print(f"Service started. Task scheduled at {schedule_time} every day.")

# 主循环，保持服务运行
os.chdir("/home/ubuntu/weibo-search-master")
while True:
    try:
        schedule.run_pending()
        if exit:
            break
        time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(60)  # 如果发生错误，等待一段时间后重试
print("exited safely")
