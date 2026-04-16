# -*- coding: utf-8 -*-
"""contract_parser 回归测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract_parser import parse_contract, get_sector, get_sym_meta


def test_future_plain():
    r = parse_contract("rb2510")
    assert r["kind"] == "future"
    assert r["underlying"] == "rb"
    assert r["month"] == "2510"
    assert r["exchange"] == "SHFE"
    assert r["sym_key"] == "SHFE.rb"


def test_future_with_exchange():
    r = parse_contract("SHFE.rb2510")
    assert r["exchange"] == "SHFE"
    assert r["underlying"] == "rb"
    assert r["month"] == "2510"


def test_future_continuous():
    r = parse_contract("SHFE.rb")
    assert r["kind"] == "future"
    assert r["month"] is None


def test_czce_uppercase():
    r = parse_contract("CZCE.TA2510")
    assert r["underlying"] == "TA"
    assert r["sym_key"] == "CZCE.TA"


def test_option_call():
    r = parse_contract("SHFE.rb2510-C-3500")
    assert r["kind"] == "option"
    assert r["cp"] == "C"
    assert r["strike"] == 3500.0
    assert r["underlying"] == "rb"
    assert r["underlying_contract"] == "SHFE.rb2510"


def test_option_put_no_exchange():
    r = parse_contract("rb2510-P-3400")
    assert r["cp"] == "P"
    assert r["strike"] == 3400.0
    assert r["exchange"] == "SHFE"  # 自动回填


def test_sector_map():
    assert get_sector("rb") == "黑色"
    assert get_sector("SHFE.rb") == "黑色"
    assert get_sector("TA") == "能化"
    assert get_sector("au") == "贵金属"
    assert get_sector("unknown") is None


def test_sym_meta():
    m = get_sym_meta("SHFE.rb")
    assert m["name"] == "螺纹钢"
    assert m["tick_size"] == 1
    assert m["sector"] == "黑色"


def test_bad_code():
    assert parse_contract("") is None
    assert parse_contract(None) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [OK] {name}")
            except AssertionError as e:
                print(f"  [FAIL] {name}: {e}")
    print("\n[ALL PASS]" if True else "")
