# -*- coding: utf-8 -*-
"""
期权品种探测脚本
================
连接天勤，遍历所有品种，探测哪些品种有对应期权合约。
输出结果表 + 保存到 probe_results.json
"""

import json
import os
import sys
from tqsdk import TqApi, TqAuth
import config as cfg

# 所有待探测品种（交易所.品种代码 → 中文名）
# 包含 SYMBOL_CONFIGS 全部品种 + 额外可能有期权的品种
PROBE_SYMBOLS = {
    # 贵金属
    "SHFE.au": "黄金",
    "SHFE.ag": "白银",
    # 有色金属
    "SHFE.cu": "铜",
    "SHFE.al": "铝",
    "SHFE.zn": "锌",
    "SHFE.ni": "镍",
    "INE.bc":  "国际铜",
    # 黑色系
    "SHFE.rb": "螺纹钢",
    "SHFE.hc": "热卷",
    "DCE.i":   "铁矿石",
    "DCE.jm":  "焦煤",
    "DCE.j":   "焦炭",
    # 能化
    "SHFE.bu": "沥青",
    "SHFE.ru": "橡胶",
    "SHFE.sp": "纸浆",
    "SHFE.fu": "燃料油",
    "INE.lu":  "低硫燃油",
    "INE.sc":  "原油",
    "DCE.eg":  "乙二醇",
    "DCE.pp":  "聚丙烯",
    "DCE.l":   "塑料",
    "DCE.v":   "PVC",
    "DCE.pg":  "LPG",
    "DCE.eb":  "苯乙烯",
    "CZCE.TA": "PTA",
    "CZCE.MA": "甲醇",
    "CZCE.SA": "纯碱",
    "CZCE.FG": "玻璃",
    # 农产品
    "DCE.m":   "豆粕",
    "DCE.y":   "豆油",
    "DCE.p":   "棕榈油",
    "DCE.lh":  "生猪",
    "CZCE.SR": "白糖",
    "CZCE.CF": "棉花",
    "CZCE.RM": "菜粕",
    # 其他
    "GFEX.lc": "碳酸锂",
}


def probe_all(api):
    """探测所有品种的期权情况"""
    results = {}

    for symbol_key, name in PROBE_SYMBOLS.items():
        exchange, product = symbol_key.split(".")

        main_contract = ""
        option_count = 0

        try:
            # Step 1: 获取主力合约
            cont_list = api.query_cont_quotes(
                exchange_id=exchange,
                product_id=product
            )
            if cont_list:
                main_contract = cont_list[0]

                # Step 2: 查询该主力合约的未到期期权
                opts = api.query_options(main_contract, expired=False)
                option_count = len(opts) if opts else 0
        except Exception as e:
            print(f"  WARNING: probe failed for {symbol_key}: {e}")

        has_options = option_count > 0
        status = "有期权" if has_options else "无期权"

        results[symbol_key] = {
            "name": name,
            "has_options": has_options,
            "main_contract": main_contract,
            "option_count": option_count,
        }

        print(f"  {symbol_key:12s} {name:6s} | {status} | 主力:{main_contract:16s} | 期权合约:{option_count:4d}")

    return results


def main():
    print("=" * 70)
    print("期权品种探测")
    print("=" * 70)

    api = TqApi(auth=TqAuth(cfg.TQ_ACCOUNT, cfg.TQ_PASSWORD))

    try:
        results = probe_all(api)

        # 分类汇总
        with_options = {k: v for k, v in results.items() if v["has_options"]}
        without_options = {k: v for k, v in results.items() if not v["has_options"]}

        print("\n" + "=" * 70)
        print(f"有期权: {len(with_options)} 个品种")
        print("-" * 40)
        for k, v in with_options.items():
            print(f"  {k:12s} {v['name']}")

        print(f"\n无期权: {len(without_options)} 个品种")
        print("-" * 40)
        for k, v in without_options.items():
            print(f"  {k:12s} {v['name']}")

        # 保存结果
        output = {
            "probe_date": "2026-04-04",
            "total_probed": len(results),
            "with_options_count": len(with_options),
            "results": results,
            "qualified_symbols": list(with_options.keys()),
        }

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {out_path}")

    finally:
        api.close()


if __name__ == "__main__":
    main()
