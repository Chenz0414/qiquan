# -*- coding: utf-8 -*-
"""
FastAPI 应用
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.routes import create_router


def create_app(dashboard_state, signal_db) -> FastAPI:
    app = FastAPI(title="期货信号监控")
    app.state.dashboard = dashboard_state
    app.state.db = signal_db

    base = os.path.dirname(__file__)
    template_dir = os.path.join(base, "templates")
    static_dir = os.path.join(base, "static")
    os.makedirs(static_dir, exist_ok=True)

    app.state.templates = Jinja2Templates(directory=template_dir)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.include_router(create_router())
    return app
