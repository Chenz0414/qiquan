/* ============================================================
   期货信号监控 · Phase C1 前端 Store + SSE
   ============================================================ */

document.addEventListener('alpine:init', () => {

  // ------------------------------------------------------------
  //  Store: 所有响应式状态都在这里
  // ------------------------------------------------------------
  Alpine.store('dash', {
    // 连接/引擎状态
    connected: false,
    engineStatus: 'unknown',
    lastUpdate: null,
    sseReconnects: 0,

    // 数据切片
    activePositions: {},      // sym_key -> position
    detectorTrends: {},
    lastBarTime: {},
    symbolStates: {},
    candidatePool: {},        // sym_key -> candidate
    symbolHeatmap: {},
    portfolioLive: {},        // signal_id -> live metrics
    sectorExposure: {},
    ruleTodayCounters: {},
    driftState: {},
    rejectStream: [],         // 最近 200
    rulesCatalog: null,

    // UI 选中态
    selectedContract: null,   // inspect 结果
    selectedRule: null,

    // Drawer (C4-C5)
    drawerOpen: true,
    drawerTab: 'heatmap',     // heatmap|sector|rejects_agg|rules_doc
    showShortcuts: false,

    // 衍生
    get candidates() {
      // 排序优先级：触发中(0) < 即将触发(1-3) < 较远(>3) < 已突破(<0) < 未知(null)
      const tier = (t) => {
        if (t == null) return 9;
        if (t === 0) return 0;
        if (t > 0 && t <= 3) return 1;
        if (t > 3) return 2;
        return 3; // t < 0 已突破
      };
      return Object.values(this.candidatePool).sort((a, b) => {
        const ta = tier(a.distance_ticks);
        const tb = tier(b.distance_ticks);
        if (ta !== tb) return ta - tb;
        // 同优先级内：|ticks| 升序
        return Math.abs(a.distance_ticks ?? 9999)
             - Math.abs(b.distance_ticks ?? 9999);
      });
    },
    get openPositions() {
      return Object.values(this.activePositions);
    },
    get openPositionCount() {
      return this.openPositions.length;
    },
    get candidateCount() {
      return Object.keys(this.candidatePool).length;
    },
    // 热力图按板块分组
    get heatmapBySector() {
      const out = {};
      for (const cell of Object.values(this.symbolHeatmap)) {
        const sec = cell.sector || '未知';
        (out[sec] = out[sec] || []).push(cell);
      }
      // 每组按 |er20| desc
      for (const sec in out) {
        out[sec].sort((a, b) =>
          Math.abs(b.er20 || 0) - Math.abs(a.er20 || 0));
      }
      return out;
    },
    // 拒绝流水按 reason 聚合（今日）
    get rejectAggregate() {
      const out = {};
      for (const r of this.rejectStream) {
        const key = r.reason || 'unknown';
        out[key] = (out[key] || 0) + 1;
      }
      return Object.entries(out)
        .sort((a, b) => b[1] - a[1])
        .map(([reason, count]) => ({ reason, count }));
    },
    // 板块警告（同向 ≥3）
    get sectorWarnings() {
      const warns = [];
      for (const [sec, slot] of Object.entries(this.sectorExposure)) {
        const longTotal = (slot.long_count || 0) + (slot.long_candidates || []).length;
        const shortTotal = (slot.short_count || 0) + (slot.short_candidates || []).length;
        if (longTotal >= 3) warns.push({ sector: sec, direction: 'long', count: longTotal });
        if (shortTotal >= 3) warns.push({ sector: sec, direction: 'short', count: shortTotal });
      }
      return warns;
    },
  });

  // ------------------------------------------------------------
  //  SSE 单例
  // ------------------------------------------------------------
  const dash = Alpine.store('dash');
  let es = null;

  function connectSSE() {
    if (es) { try { es.close(); } catch(_) {} }
    es = new EventSource('/api/events');

    es.addEventListener('connected', () => {
      dash.connected = true;
      console.log('[SSE] connected');
    });

    es.addEventListener('heartbeat', () => { /* keepalive */ });

    // 老事件（兼容）
    es.addEventListener('new_signal', () => loadSnapshot());
    es.addEventListener('stop_update', () => loadSnapshot());
    es.addEventListener('position_closed', () => loadSnapshot());
    es.addEventListener('status_change', (e) => {
      const d = safeJson(e.data);
      if (d && d.status) dash.engineStatus = d.status;
    });

    // B5 新事件
    es.addEventListener('candidate_update', (e) => {
      const d = safeJson(e.data); if (!d) return;
      (d.upserts || []).forEach(c => {
        if (c && c.sym_key) dash.candidatePool[c.sym_key] = c;
      });
      (d.removes || []).forEach(k => { delete dash.candidatePool[k]; });
    });

    es.addEventListener('heatmap_delta', (e) => {
      const d = safeJson(e.data); if (!d) return;
      (d.cells || []).forEach(c => {
        if (c && c.sym_key) dash.symbolHeatmap[c.sym_key] = c;
      });
    });

    es.addEventListener('position_live', (e) => {
      const d = safeJson(e.data); if (!d) return;
      if (d.removed) {
        delete dash.portfolioLive[d.signal_id];
      } else if (d.live) {
        dash.portfolioLive[d.signal_id] = d.live;
      }
    });

    es.addEventListener('reject', (e) => {
      const d = safeJson(e.data); if (!d) return;
      dash.rejectStream.unshift(d);
      if (dash.rejectStream.length > 200) dash.rejectStream.length = 200;
    });

    es.addEventListener('sector_warning', (e) => {
      const d = safeJson(e.data); if (!d) return;
      console.warn('[SECTOR WARNING]', d);
    });

    es.addEventListener('drift_alert', (e) => {
      const d = safeJson(e.data); if (!d) return;
      if (d.rule_key) dash.driftState[d.rule_key] = d;
    });

    es.onerror = () => {
      dash.connected = false;
      dash.sseReconnects += 1;
      console.warn('[SSE] disconnected, reconnect in 3s');
      setTimeout(connectSSE, 3000);
    };
  }

  function safeJson(s) {
    try { return JSON.parse(s); } catch(_) { return null; }
  }

  // ------------------------------------------------------------
  //  首加载：REST 拉全量，之后 SSE 增量
  // ------------------------------------------------------------
  async function loadSnapshot() {
    try {
      const r = await fetch('/api/snapshot');
      if (!r.ok) throw new Error('snapshot ' + r.status);
      const snap = await r.json();
      dash.engineStatus = snap.engine_status || 'unknown';
      dash.activePositions = snap.active_positions || {};
      dash.detectorTrends = snap.detector_trends || {};
      dash.lastBarTime = snap.last_bar_time || {};
      dash.symbolStates = snap.symbol_states || {};
      dash.candidatePool = snap.candidate_pool || {};
      dash.symbolHeatmap = snap.symbol_heatmap || {};
      dash.portfolioLive = snap.portfolio_live || {};
      dash.sectorExposure = snap.sector_exposure || {};
      dash.ruleTodayCounters = snap.rule_today_counters || {};
      dash.driftState = snap.drift_state || {};
      dash.rejectStream = snap.reject_stream || [];
      dash.lastUpdate = new Date().toISOString();
    } catch(e) {
      console.error('[snapshot]', e);
    }
  }

  async function loadRulesCatalog() {
    try {
      const r = await fetch('/api/rules/catalog');
      if (!r.ok) return;
      dash.rulesCatalog = await r.json();
    } catch(e) { console.error('[catalog]', e); }
  }

  // ------------------------------------------------------------
  //  暴露给组件的 API
  // ------------------------------------------------------------
  window.dashApi = {
    async inspect(code) {
      if (!code) return;
      try {
        const r = await fetch('/api/contract/inspect?code=' + encodeURIComponent(code));
        if (!r.ok) throw new Error('inspect ' + r.status);
        dash.selectedContract = await r.json();
      } catch(e) {
        dash.selectedContract = { error: String(e) };
      }
    },
    clearInspect() { dash.selectedContract = null; },
  };

  // ------------------------------------------------------------
  //  C8 全局快捷键
  // ------------------------------------------------------------
  document.addEventListener('keydown', (e) => {
    // 在 input/textarea 内不拦截
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') return;

    if (e.key === '/') {
      e.preventDefault();
      const input = document.querySelector('input[data-role="inspect"]');
      if (input) input.focus();
    } else if (e.key === 'h') {
      dash.drawerOpen = true;
      dash.drawerTab = 'heatmap';
    } else if (e.key === 's') {
      dash.drawerOpen = true;
      dash.drawerTab = 'sector';
    } else if (e.key === 'j') {
      dash.drawerOpen = true;
      dash.drawerTab = 'rejects_agg';
    } else if (e.key === 'r') {
      dash.drawerOpen = true;
      dash.drawerTab = 'rules_doc';
    } else if (e.key === 'd') {
      dash.drawerOpen = !dash.drawerOpen;
    } else if (e.key === '?') {
      dash.showShortcuts = !dash.showShortcuts;
    } else if (e.key === 'Escape') {
      dash.showShortcuts = false;
    }
  });

  // 启动
  loadRulesCatalog();
  loadSnapshot();
  connectSSE();
  // 兜底刷新：60s
  setInterval(loadSnapshot, 60000);
});

// ------------------------------------------------------------
//  纯工具函数（非响应式）
// ------------------------------------------------------------
window.fmt = {
  time(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    if (isNaN(d)) return '--';
    return String(d.getHours()).padStart(2, '0') + ':' +
           String(d.getMinutes()).padStart(2, '0') + ':' +
           String(d.getSeconds()).padStart(2, '0');
  },
  dateTime(iso) {
    if (!iso) return '--';
    return String(iso).slice(0, 16).replace('T', ' ');
  },
  num(v, digits=2) {
    if (v == null || isNaN(v)) return '--';
    return Number(v).toFixed(digits);
  },
  pct(v, digits=1) {
    if (v == null || isNaN(v)) return '--';
    const sign = v >= 0 ? '+' : '';
    return sign + Number(v).toFixed(digits) + '%';
  },
  r(v, digits=2) {
    if (v == null || isNaN(v)) return '--';
    const sign = v >= 0 ? '+' : '';
    return sign + Number(v).toFixed(digits) + 'R';
  },
  distClass(ticks) {
    if (ticks == null) return '';
    if (ticks < 0) return 'past';
    if (ticks === 0) return 'hit';
    if (ticks <= 3) return 'near';
    return '';
  },
};
