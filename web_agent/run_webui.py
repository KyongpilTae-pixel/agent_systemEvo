"""Open WebUI 기동 래퍼 (env openwebui 의 python 으로 실행).

★pyarrow 를 가장 먼저 로드한다. open_webui.main 이 langchain/scipy/torch 등을 먼저 불러온 뒤 pyarrow.lib 를
로드하면 순서 의존 네이티브 충돌로 세그폴트(139)가 난다 (2026-09-30, open-webui 0.11.4 / pyarrow 20.0.0).
단독 import 는 정상이라 import 순서만 바꿔 회피한다.
"""
import pyarrow.lib  # noqa: F401  — 반드시 첫 import
import pandas  # noqa: F401

import os
import uvicorn

uvicorn.run("open_webui.main:app", host=os.environ.get("WEBUI_HOST", "127.0.0.1"),
            port=int(os.environ.get("WEBUI_PORT", "8300")))
