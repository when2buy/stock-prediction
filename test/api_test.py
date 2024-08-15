from random import choice
from json import dumps

from locust import task
from locust import HttpUser

from util.conf import get_conf

conf = get_conf()

examples = [
    "我想要查询一下北京的天气",
    "我想要查询一下上海的天气",
    "我想要查询一下广州的天气",
    "我想要查询一下深圳的天气",
    "我想要查询一下武汉的天气",
    "我想要查询一下杭州的天气",
    "我想要查询一下南京的天气",
    "我想要查询一下成都的天气",
    "我想要查询一下重庆的天气",
]


class MyUser(HttpUser):
    host = "http://" + conf["server"]["host"] + ":" + str(conf["server"]["port"])

    @task()
    def query_test(self):
        header = {"Content-Type": "application/json"}
        data = {"text": choice(examples)}
        self.client.post("/test", data=dumps(data), headers=header, verify=False)
