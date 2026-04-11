# 期货信号监控 + 回测研究系统

## 项目概述

均值回归期货交易系统，两种运行模式：
- **实时监控**：云服务器(腾讯云) 运行 `run_monitor.py`，检测信号并推送微信
- **离线回测**：本地 Windows 运行 `test_*.py` / `backtest_engine.py`，研究策略

## 三个环境

| 环境 | 位置 | 用途 |
|------|------|------|
| 本地 | `C:\Users\Administrator\Desktop\qiquan` | 回测研究 + 开发 |
| GitHub | `Chenz0414/qiquan.git` (origin/main) | 代码同步中转 |
| 腾讯云 | `/home/monitor/qiquan` | systemd 运行监控服务 |

同步流程：本地开发 → `git push` → 云服务器 `git pull` + `systemctl restart monitor`

## 文件分类

### 核心共享模块（改动需注意两端兼容）
- `signal_core.py` — 信号检测 + 出场追踪（A/B/C信号、8种出场策略、场景分类）
- `config.py` — 回测配置（品种/周期/均线参数）
- `data_loader.py` — 数据加载 + 指标计算（EMA/ER/ATR）
- `data_cache.py` — K线缓存（parquet格式，永久有效）
- `stats_utils.py` — EV/胜率/过滤等统计工具

### 监控专用（云服务器运行）
- `monitor.py` — 实时监控引擎（TqSdk订阅 + 信号检测循环）
- `monitor_config.py` — 监控配置（从 monitor_settings.json 加载）
- `notifier.py` — 微信推送（PushPlus）
- `run_monitor.py` — CLI入口
- `state_manager.py` — 状态持久化（断线恢复用）
- `signal_db.py` — SQLite交易记录
- `web/` — FastAPI仪表盘（SSE实时更新）

### 回测专用（本地运行）
- `backtest_engine.py` — 离线信号模拟
- `analysis.py` — 回测统计报告
- `chart_engine.py` — K线图渲染（Canvas）
- `report_engine.py` — HTML报告生成
- `option_pricing.py` — Black-76期权定价
- `volatility.py` — 历史波动率

### 研究脚本（一次性/探索用）
- `test_*.py` — 80+个回测研究脚本
- `scan_*.py` / `phase_analysis*.py` — 特定分析工具
- `demo_toolkit.py` — 工具链使用示例

### 部署配置
- `deploy/monitor.service` — systemd服务定义
- `deploy/nginx.conf` — 反向代理
- `deploy/setup.sh` — 云服务器初始化

## 开发规范

### 修改 signal_core.py 时
这是监控和回测共享的核心，改动影响两端：
1. 改完先本地跑回测验证（`test_s6_robustness.py` 或相关 test）
2. 确认无误后 push，再到云服务器 pull + restart
3. 新增出场策略：在 `ExitTracker` 加 `sX_*` 字段，同步更新 `SCENARIO_EXIT` 映射

### 凭证管理
- **禁止**在代码中硬编码新的账号密码
- `monitor_settings.json` 已在 .gitignore，云服务器单独维护
- `config.py` 中的天勤账号仅供本地回测，不要改成别的敏感信息
- 如需新的密钥/token，用环境变量或本地 json 文件，加入 .gitignore

### 数据缓存
- 缓存目录：`data_cache/`（已 gitignore）
- 格式：`{交易所}_{品种}_{周期}min_{天数}d.parquet`
- 永久有效，需要刷新时用 `batch_pull.py` 或 `data_cache.get_klines(force=True)`
- 回测前问用户：用缓存还是重新拉取

### 回测输出
- 输出目录：`output/`（已 gitignore）
- 回测报告必须配K线图HTML验证（用 chart_engine + report_engine）
- 关注 MFE 而非固定百分比盈亏，关注爆发率

### Git 提交
- 研究脚本（test_*.py）可以随意提交，不影响生产
- 核心模块改动必须在提交消息中说明影响范围
- 不要提交：`data_cache/`, `output/`, `state/`, `logs/`, `*.db`, `monitor_settings.json`

## 云服务器部署检查单

```bash
# 1. 拉取最新代码
cd /home/monitor/qiquan && git pull

# 2. 如有新依赖
source /home/monitor/venv/bin/activate && pip install -r requirements.txt

# 3. 重启服务
sudo systemctl restart monitor

# 4. 确认运行
sudo systemctl status monitor
journalctl -u monitor -f --no-pager -n 50
```

## 关键参数速查

| 参数 | 值 | 位置 |
|------|-----|------|
| EMA信号线 | 10 | config.py |
| EMA趋势线 | 20 | config.py |
| EMA长期线 | 120 | config.py |
| ATR周期 | 14 | config.py |
| ER门槛(入场) | ≥0.5 | signal_core.py |
| ER门槛(过滤) | ≥0.7禁入 | signal_core.py |
| 止损缓冲 | 5跳 | config.py |
| 推送冷却 | 300秒 | monitor_settings.json |
| 回测天数 | 120天 | config.py |
| 预热天数 | 50天 | config.py |
| 品种总数 | 32个 | signal_core.SYMBOL_CONFIGS |

## 场景 → 出场映射（当前版本）

| 场景 | 条件 | 出场策略 |
|------|------|---------|
| 场景1 | A信号 + ER≥0.5 + 偏离≥1.0ATR | S6 |
| 场景2 | C信号 + 偏离≥2.0ATR | S6 |
| 场景3 | B信号 + ER≥0.5 + 0.1≤偏离<0.3ATR | S5.1 |
