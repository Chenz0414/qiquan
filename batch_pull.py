# -*- coding: utf-8 -*-
"""
批量数据拉取脚本
================
根据 probe_results.json 中有期权的品种，拉取 2min + 10min K线数据。
使用 data_cache.get_klines() 拉取并缓存。

用法:
    python batch_pull.py              # 拉取全部有期权品种
    python batch_pull.py --dry-run    # 只打印计划，不实际拉取
"""

import json
import os
import sys
import time
from tqsdk import TqApi, TqAuth
import config as cfg
from data_cache import get_klines, clear_cache

PERIODS = [2, 10]       # 分钟
DAYS = 170              # 请求天数

PROBE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results.json")


def load_qualified_symbols():
    """从 probe_results.json 加载有期权的品种列表"""
    if not os.path.exists(PROBE_FILE):
        print(f"ERROR: {PROBE_FILE} 不存在，请先运行 probe_options.py")
        sys.exit(1)

    with open(PROBE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    symbols = data.get("qualified_symbols", [])
    results = data.get("results", {})

    print(f"从 probe_results.json 加载 {len(symbols)} 个有期权品种")
    return symbols, results


def build_tq_symbol(symbol_key):
    """SHFE.au → KQ.m@SHFE.au"""
    return f"KQ.m@{symbol_key}"


def main():
    dry_run = "--dry-run" in sys.argv

    symbols, results = load_qualified_symbols()
    total_tasks = len(symbols) * len(PERIODS)

    print("=" * 70)
    print(f"批量数据拉取: {len(symbols)} 品种 × {len(PERIODS)} 周期 = {total_tasks} 次拉取")
    print(f"周期: {PERIODS}")
    print(f"天数: {DAYS}")
    print("=" * 70)

    if dry_run:
        print("\n[DRY RUN] 拉取计划:")
        for i, sym in enumerate(symbols, 1):
            name = results.get(sym, {}).get("name", "")
            for p in PERIODS:
                print(f"  [{i}] {sym} ({name}) {p}min")
        print(f"\n共 {total_tasks} 次拉取")
        return

    # 先清除旧缓存
    print("\n清除旧缓存...")
    cleared = clear_cache()
    print(f"已清除 {cleared} 个旧缓存文件\n")

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    try:
        done = 0
        failed = []
        summary = []

        for sym_key in symbols:
            name = results.get(sym_key, {}).get("name", sym_key)
            tq_sym = build_tq_symbol(sym_key)

            for period in PERIODS:
                done += 1
                print(f"\n[{done}/{total_tasks}] {sym_key} ({name}) {period}min")

                try:
                    df = get_klines(api, tq_sym, sym_key,
                                    period_min=period, days=DAYS,
                                    force_refresh=True)
                    row_count = len(df)
                    date_range = ""
                    if row_count > 0:
                        d0 = df['datetime'].iloc[0].strftime('%Y-%m-%d')
                        d1 = df['datetime'].iloc[-1].strftime('%Y-%m-%d')
                        date_range = f"{d0} ~ {d1}"

                    summary.append({
                        "symbol": sym_key,
                        "name": name,
                        "period": period,
                        "rows": row_count,
                        "date_range": date_range,
                        "status": "OK"
                    })
                    print(f"  OK: {row_count} bars, {date_range}")

                except Exception as e:
                    failed.append(f"{sym_key} {period}min: {e}")
                    summary.append({
                        "symbol": sym_key,
                        "name": name,
                        "period": period,
                        "rows": 0,
                        "date_range": "",
                        "status": f"FAIL: {e}"
                    })
                    print(f"  FAIL: {e}")

                time.sleep(0.5)

        # 汇总报告
        print("\n" + "=" * 70)
        print("拉取完成汇总")
        print("=" * 70)
        print(f"{'品种':12s} {'名称':6s} {'周期':>5s} {'K线数':>7s} {'日期范围':20s} {'状态'}")
        print("-" * 70)
        for s in summary:
            print(f"{s['symbol']:12s} {s['name']:6s} {s['period']:4d}m {s['rows']:7d} {s['date_range']:20s} {s['status']}")

        ok_count = sum(1 for s in summary if s['status'] == 'OK')
        print(f"\n成功: {ok_count}/{total_tasks}")
        if failed:
            print(f"失败: {len(failed)}")
            for f in failed:
                print(f"  - {f}")

    finally:
        api.close()


if __name__ == "__main__":
    main()
