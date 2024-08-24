from glm import GLM 
from weibo_processor import WeiboProcessor
# Database configuration
db_config = {
    'user': 'weibo',
    'password': '123456',
    'host': '175.27.156.150',
    'database': 'weibo_data',
    'raise_on_warnings': True
}

glm = GLM()
processor = WeiboProcessor(db_config, glm)
processor.run(1000)
