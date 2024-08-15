from asyncio import sleep

from sanic import Sanic
from sanic.response import text, json

from util.conf import get_conf
from util.data_model import Data

conf = get_conf()
NAME = conf["server"]["name"]
VERSION = conf["server"]["version"]
app = Sanic(NAME)


@app.get("/")
async def index(_):
    return text(f"Hello from {NAME}: {VERSION}")


@app.get("/version")
async def version(_):
    return json({
        "name": NAME,
        "version": VERSION
    })


@app.post("/test")
async def test(request):
    print(request.json)
    try:
        data = Data(**request.json)
        await sleep(0.2)
        result = {
            "status": 1,
            "message": "success",
            "data": data.dict()
        }
    except Exception as e:
        result = {
            "status": 0,
            "message": str(e)
        }
    return json(result)


if __name__ == '__main__':
    app.run(
        host=conf["server"]["host"],
        port=conf["server"]["port"],
        debug=conf["server"]["debug"],
        auto_reload=conf["server"]["auto_reload"]
    )
