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
