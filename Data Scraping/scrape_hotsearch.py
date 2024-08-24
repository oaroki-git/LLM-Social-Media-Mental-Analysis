#引入requests库
import requests
# 获取json文件

def hot_search():
    url = 'https://weibo.com/ajax/side/hotSearch'
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()['data']

def run(num):
    open("/home/ubuntu/weibo-search-master/keywords.txt", "w").close() 
    keywordfile = open("/home/ubuntu/weibo-search-master/keywords.txt", "a") 
    data = hot_search()
    if not data:
        print('获取微博热搜榜失败')
        return
    for i, rs in enumerate(data['realtime'][:num], 1):
        title = rs['word']
        keywordfile.write(f"#{title}#\n")
    keywordfile.close()
