# -*- coding: utf-8 -*-
"""快速测试天勤连接"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from tqsdk import TqApi, TqAuth

print("connecting...")
api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))
print("connected, fetching 10 bars of ag...")
klines = api.get_kline_serial("KQ.m@SHFE.ag", duration_seconds=600, data_length=10)
api.wait_update()
print(f"got {len(klines)} bars")
print(klines[['datetime','close']].tail(3))
api.close()
print("done")
