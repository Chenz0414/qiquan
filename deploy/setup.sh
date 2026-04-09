#!/bin/bash
# 腾讯云服务器初始化脚本
# 运行: sudo bash setup.sh

set -e

echo "=== 1. 系统更新 ==="
apt update && apt upgrade -y
apt install -y python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx

echo "=== 2. 创建用户 ==="
useradd -m -s /bin/bash monitor || true

echo "=== 3. Python环境 ==="
su - monitor -c "
python3.11 -m venv /home/monitor/venv
source /home/monitor/venv/bin/activate
pip install tqsdk fastapi uvicorn jinja2 requests pandas numpy
"

echo "=== 4. 上传代码后，配置 nginx ==="
echo "   cp deploy/nginx.conf /etc/nginx/sites-available/monitor"
echo "   ln -s /etc/nginx/sites-available/monitor /etc/nginx/sites-enabled/"
echo "   nginx -t && systemctl reload nginx"

echo "=== 5. 配置 HTTPS ==="
echo "   certbot --nginx -d your-domain.com"

echo "=== 6. 启动服务 ==="
echo "   cp deploy/monitor.service /etc/systemd/system/"
echo "   systemctl daemon-reload"
echo "   systemctl enable monitor"
echo "   systemctl start monitor"

echo "=== 完成！==="
echo "查看日志: journalctl -u monitor -f"
