# -*- coding: utf-8 -*-
"""
HTML 报告引擎
=============
组装 K 线图、统计表格、文本段落 生成完整 HTML 报告。

用法:
    from report_engine import Report

    rpt = Report('ER(20) 研究')
    rpt.add_section('全品种 S2')
    rpt.add_table(headers, rows)
    rpt.add_chart(chart_html)          # chart_engine.render_chart() 的返回值
    rpt.save('output/my_report.html')
"""

import os
from chart_engine import get_chart_js


class Report:
    """HTML 报告构建器"""

    def __init__(self, title='研究报告'):
        self.title = title
        self._parts = []

    def add_section(self, title, subtitle=None):
        """添加章节标题"""
        html = f'<h2>{title}</h2>'
        if subtitle:
            html += f'<p class="subtitle">{subtitle}</p>'
        self._parts.append(html)

    def add_text(self, text, color=None):
        """添加文本段落"""
        style = f' style="color:{color}"' if color else ''
        self._parts.append(f'<p{style}>{text}</p>')

    def add_table(self, headers, rows, highlight_pnl_cols=None):
        """
        添加数据表格。

        参数:
          headers: ['列名1', '列名2', ...]
          rows: [[val1, val2, ...], ...] — 值可以是 str/int/float
          highlight_pnl_cols: 哪些列索引要做盈亏着色（正绿负红），如 [3, 5, 7]
        """
        pnl_set = set(highlight_pnl_cols or [])
        lines = ['<table>']
        # 表头
        lines.append('<tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>')
        # 行
        for row in rows:
            cells = []
            for ci, val in enumerate(row):
                css = ''
                if ci in pnl_set:
                    try:
                        fv = float(val)
                        if fv > 0:
                            css = ' class="pos"'
                        elif fv < 0:
                            css = ' class="neg"'
                    except (ValueError, TypeError):
                        pass
                cells.append(f'<td{css}>{val}</td>')
            lines.append('<tr>' + ''.join(cells) + '</tr>')
        lines.append('</table>')
        self._parts.append('\n'.join(lines))

    def add_ev_table(self, title, ev_rows, strategies=('s1', 's2', 's3', 's4')):
        """
        添加 EV 统计表（常用格式：ER分档 × 出场策略）。

        参数:
          title: 表格标题
          ev_rows: [{'er': '0.5~0.6', 's1': {N, EV, wr, pr, ...}, 's2': {...}, ...}, ...]
          strategies: 要显示的出场策略
        """
        headers = ['ER档']
        for sx in strategies:
            headers.extend([f'{sx.upper()} N', f'{sx.upper()} EV', f'{sx.upper()} WR%',
                            f'{sx.upper()} PR', f'{sx.upper()} Σ'])

        rows = []
        pnl_cols = []
        for si, sx in enumerate(strategies):
            base = 1 + si * 5
            pnl_cols.extend([base + 1, base + 4])  # EV 和 sum 列

        for r in ev_rows:
            row = [r['er']]
            for sx in strategies:
                st = r.get(sx, {})
                row.extend([
                    st.get('N', 0),
                    st.get('EV', 0),
                    st.get('wr', 0),
                    st.get('pr', 0),
                    st.get('sum_pnl', 0),
                ])
            rows.append(row)

        self.add_section(title)
        self.add_table(headers, rows, highlight_pnl_cols=pnl_cols)

    def add_chart(self, chart_html):
        """添加单个图表（chart_engine.render_chart 的返回值）"""
        self._parts.append(chart_html)

    def add_html(self, raw_html):
        """添加任意 HTML 片段"""
        self._parts.append(raw_html)

    def to_html(self):
        """生成完整 HTML 字符串"""
        body = '\n'.join(self._parts)
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{self.title}</title>
<script>
{get_chart_js()}
</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0d1117;color:#c9d1d9;font-family:'Microsoft YaHei','Consolas',sans-serif;
  padding:20px;max-width:1600px;margin:0 auto}}
h1{{color:#58a6ff;font-size:22px;margin-bottom:8px;border-bottom:2px solid #21262d;padding-bottom:8px}}
h2{{color:#ff7b72;font-size:17px;margin:28px 0 8px;border-bottom:1px solid #30363d;padding-bottom:5px}}
.subtitle{{color:#8b949e;font-size:12px;margin-bottom:8px}}
p{{margin:6px 0;font-size:13px;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 16px}}
th{{background:#161b22;color:#8b949e;padding:7px 8px;text-align:right;border:1px solid #21262d;
  position:sticky;top:0;white-space:nowrap}}
td{{padding:6px 8px;text-align:right;border:1px solid #21262d}}
th:first-child,td:first-child{{text-align:left}}
tr:hover td{{background:#1c2333}}
.pos{{color:#3fb950}} .neg{{color:#f85149}}
.chart-box{{background:#161b22;border-radius:8px;padding:10px;margin:10px 0;
  border:1px solid #30363d}}
canvas{{width:100%;display:block}}
</style>
</head>
<body>
<h1>{self.title}</h1>
{body}
</body>
</html>"""

    def save(self, path):
        """保存 HTML 到文件"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_html())
        print(f"Report saved: {path}")
