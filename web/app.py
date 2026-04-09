# -*- coding: utf-8 -*-
"""
FastAPI 应用
"""

import os
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from web.routes import create_router


def create_app(dashboard_state, signal_db) -> FastAPI:
    app = FastAPI(title="期货信号监控")
    app.state.dashboard = dashboard_state
    app.state.db = signal_db

    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app.state.templates = Jinja2Templates(directory=template_dir)

    app.include_router(create_router())
    return app
