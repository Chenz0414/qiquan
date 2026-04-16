# 部署与同步指南

## 日常更新流程

### 本地改完代码后

```bash
# 1. 本地提交并推送
git add <改动的文件>
git commit -m "描述改动"
git push

# 2. SSH 到云服务器
ssh -i qiquan.pem ubuntu@124.222.55.3

# 3. 拉取并重启
cd /home/ubuntu/qiquan
git pull
sudo systemctl restart monitor

# 4. 确认运行正常
sudo systemctl status monitor
journalctl -u monitor -f --no-pager -n 30
```

### 如果有新依赖

```bash
source /home/ubuntu/qiquan/venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart monitor
```

## 云服务器配置文件

以下文件在云服务器**单独维护**，不走 git 同步：

| 文件 | 位置 | 说明 |
|------|------|------|
| `monitor_settings.json` | `/home/ubuntu/qiquan/` | 天勤账号 + PushPlus token |
| `.secrets.json` | `/home/ubuntu/qiquan/` | 天勤账号（config.py 读取） |

首次部署时需手动创建这两个文件：

```bash
# monitor_settings.json
cp monitor_settings.example.json monitor_settings.json
# 编辑填入实际账号密码
nano monitor_settings.json

# .secrets.json
cat > .secrets.json << 'EOF'
{
    "tq_account": "你的账号",
    "tq_password": "你的密码"
}
EOF
```

## 哪些改动需要重启服务

| 改了什么 | 需要重启？ | 风险 |
|---------|-----------|------|
| `test_*.py` 回测脚本 | 不需要 | 无 |
| `chart_engine.py` / `report_engine.py` | 不需要 | 无（监控不用） |
| `signal_core.py` | **需要** | **高** — 影响信号检测逻辑 |
| `monitor.py` | **需要** | **高** — 监控引擎主体 |
| `notifier.py` | **需要** | 中 — 影响推送格式 |
| `config.py` | **需要** | 低 — 监控主要读 monitor_config |
| `web/*` | **需要** | 低 — 仅影响仪表盘 |
| `data_loader.py` | **需要** | 中 — 指标计算变化 |

## 重启注意事项

- 重启**不会**丢失交易记录（存在 `state/signals.db`）
- 重启**不会**丢失检测状态（`state/monitor_state.json` 会自动恢复）
- 如果改了 `SignalDetector` 的状态结构，可能需要 `--reset-state` 冷启动：
  ```bash
  sudo systemctl stop monitor
  # 编辑 ExecStart 临时加 --reset-state，或手动运行：
  cd /home/ubuntu/qiquan
  /home/ubuntu/qiquan/venv/bin/python run_monitor.py --reset-state --config monitor_settings.json
  ```

## 故障排查

```bash
# 查看服务状态
sudo systemctl status monitor

# 查看实时日志
journalctl -u monitor -f

# 查看最近错误
journalctl -u monitor --since "1 hour ago" --no-pager | grep -i error

# 手动运行（debug模式）
cd /home/ubuntu/qiquan
source /home/ubuntu/qiquan/venv/bin/activate
python run_monitor.py --config monitor_settings.json --dry-run

# 检查天勤连接
python -c "from tqsdk import TqApi, TqAuth; import json; s=json.load(open('.secrets.json')); api=TqApi(auth=TqAuth(s['tq_account'],s['tq_password'])); print('OK'); api.close()"
```

## 备份

需要备份的数据（不在 git 中）：
- `state/signals.db` — 全部交易记录
- `state/monitor_state.json` — 当前监控状态
- `monitor_settings.json` — 配置
- `logs/` — 运行日志

## Phase C-D Web 仪表盘部署

### 静态资源
新版仪表盘所有 JS/CSS 本地化在 `web/static/`：
- `alpine.min.js` / `charts.js` / `dashboard.js` / `dashboard.css`

FastAPI 已通过 `StaticFiles` 挂载 `/static/*`。旧页面保留在 `/legacy`。

### Nginx SSE 配置
如使用 nginx 反向代理，**必须**关闭缓冲，否则事件流会被卡住：

```nginx
location /api/events {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_buffering off;            # ← 关键
    proxy_cache off;
    proxy_read_timeout 24h;
    proxy_set_header Connection '';
    chunked_transfer_encoding off;
}
```

### D1-D2 定时任务
每日收盘后计算规则统计 + 生成日报并推送 PushPlus：

```bash
crontab -e
# 日盘 15:15 生成日报并推送
15 15 * * 1-5 cd /home/ubuntu/qiquan && /home/ubuntu/qiquan/venv/bin/python -m scripts.generate_daily_report --push --db state/signals.db >> logs/daily_report.log 2>&1
# 夜盘完 16:30 计算规则统计 + 漂移
30 16 * * 1-5 cd /home/ubuntu/qiquan && /home/ubuntu/qiquan/venv/bin/python -m scripts.compute_daily_rule_stats --db state/signals.db >> logs/rule_stats.log 2>&1
```

### D4 一键操作端点
- `POST /api/actions/silence` `{sym_key, minutes, reason}` — 静音品种
- `POST /api/actions/pause_rule` `{rule_key, until|"clear", reason}` — 暂停规则
- `POST /api/actions/note` `{text, signal_id?, sym_key?}` — 添加笔记
- `POST /api/actions/manual_close` `{signal_id, price, reason}` — 手动平仓
- `GET /api/actions/notes?signal_id=&limit=` — 查笔记
- `GET /api/actions/state` — 当前静音/暂停快照

### D3 沙盒回测端点
- `POST /api/sandbox/run` `{scenario?, er_min?, deviation_min?, days}` — 提交沙盒
- `GET /api/sandbox/job/:id` — 查询结果
- `GET /api/sandbox/jobs?limit=` — 历史列表

### Web 配置
启动 FastAPI 作为 systemd 服务（参考 `deploy/nginx.conf`）。
开发模式在本地用：`python -m web.dev_server` (端口 8765，带假数据)。
