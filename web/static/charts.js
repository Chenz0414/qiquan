/* ============================================================
   Canvas 绘图工具（sparkline / 热力图 cell / IV smile）
   ============================================================ */

window.charts = {

  /**
   * 绘制价格 sparkline
   * @param {CanvasRenderingContext2D} ctx
   * @param {number[]} data
   * @param {object} opts {w,h,color,thick}
   */
  sparkline(ctx, data, opts = {}) {
    const { w, h } = opts;
    const color = opts.color || '#58a6ff';
    const thick = opts.thick || 1.2;
    if (!data || data.length < 2) return;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = color;
    ctx.lineWidth = thick;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (data.length - 1)) * (w - 2) + 1;
      const y = h - 2 - ((v - min) / range) * (h - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // 最后一点
    const lastX = w - 1.5;
    const lastY = h - 2 - ((data[data.length - 1] - min) / range) * (h - 4);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 1.5, 0, Math.PI * 2);
    ctx.fill();
  },

  /**
   * 热力图格子：画边框 + 品种名 + 数据 + sparkline
   */
  heatCell(canvas, cell) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const bg = cell.trend_dir === 1 ? 'rgba(63,185,80,0.10)' :
               cell.trend_dir === -1 ? 'rgba(248,81,73,0.10)' :
               'rgba(139,148,158,0.06)';
    const lineColor = cell.trend_dir === 1 ? '#3fb950' :
                      cell.trend_dir === -1 ? '#f85149' : '#8b949e';

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);

    // 右边 sparkline（占右 60%）
    const spW = Math.floor(w * 0.55);
    const spX = w - spW - 4;
    const spH = Math.floor(h * 0.6);
    const spY = Math.floor(h * 0.2);

    if (cell.sparkline_20 && cell.sparkline_20.length >= 2) {
      ctx.save();
      ctx.translate(spX, spY);
      this.sparkline(ctx, cell.sparkline_20,
        { w: spW, h: spH, color: lineColor, thick: 1.2 });
      ctx.restore();
    }
  },

  /**
   * IV Smile 曲线：X 行权价 Y 隐含波动率，CALL/PUT 两条
   */
  ivSmile(canvas, data) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const strikes = data.strikes || [];
    const callIv = data.call_iv || [];
    const putIv = data.put_iv || [];
    if (strikes.length < 2) return;

    const all = [...callIv, ...putIv].filter(v => v != null);
    if (all.length === 0) return;
    const minIv = Math.min(...all);
    const maxIv = Math.max(...all);
    const rangeIv = (maxIv - minIv) || 0.05;
    const minK = Math.min(...strikes);
    const maxK = Math.max(...strikes);
    const rangeK = (maxK - minK) || 1;

    const pad = 24;
    const innerW = W - pad * 2;
    const innerH = H - pad * 2;

    // 坐标轴
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, innerW, innerH);

    // 画线助手
    const drawLine = (series, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let first = true;
      series.forEach((v, i) => {
        if (v == null) return;
        const x = pad + ((strikes[i] - minK) / rangeK) * innerW;
        const y = pad + innerH - ((v - minIv) / rangeIv) * innerH;
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawLine(callIv, '#3fb950');
    drawLine(putIv, '#f85149');

    // ATM 标注
    if (data.atm != null) {
      const x = pad + ((data.atm - minK) / rangeK) * innerW;
      ctx.strokeStyle = '#58a6ff';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, pad + innerH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 图例
    ctx.fillStyle = '#3fb950';
    ctx.fillText('CALL', W - 60, 16);
    ctx.fillStyle = '#f85149';
    ctx.fillText('PUT', W - 30, 16);
  },

  /**
   * 止损轨迹 mini-chart
   * @param {CanvasRenderingContext2D} ctx
   * @param {object} data {klines:[{time,close}], stop_history:[{time,new}], entry, current_price}
   */
  trajectory(canvas, data) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const klines = data.klines || [];
    if (klines.length < 2) return;

    const prices = klines.map(k => k.close);
    const stops = (data.stop_history || []).map(s => s.new);
    const all = [...prices, ...stops];
    if (data.entry != null) all.push(data.entry);
    if (data.current_price != null) all.push(data.current_price);
    const min = Math.min(...all);
    const max = Math.max(...all);
    const rng = (max - min) || 1;
    const pad = 6;
    const innerW = W - pad * 2;
    const innerH = H - pad * 2;

    // K线 close 折线
    ctx.strokeStyle = '#8b949e';
    ctx.lineWidth = 1;
    ctx.beginPath();
    klines.forEach((k, i) => {
      const x = pad + (i / (klines.length - 1)) * innerW;
      const y = pad + innerH - ((k.close - min) / rng) * innerH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 入场线（蓝虚线）
    if (data.entry != null) {
      const y = pad + innerH - ((data.entry - min) / rng) * innerH;
      ctx.strokeStyle = '#58a6ff';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + innerW, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 止损移动点（黄点）
    (data.stop_history || []).forEach(h => {
      const idx = klines.findIndex(k => k.time >= h.time);
      const i = idx < 0 ? klines.length - 1 : idx;
      const x = pad + (i / (klines.length - 1)) * innerW;
      const y = pad + innerH - ((h.new - min) / rng) * innerH;
      ctx.fillStyle = '#d29922';
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // 当前止损（红线）
    if (stops.length > 0) {
      const last = stops[stops.length - 1];
      const y = pad + innerH - ((last - min) / rng) * innerH;
      ctx.strokeStyle = '#f85149';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + innerW, y);
      ctx.stroke();
    }
  },
};
