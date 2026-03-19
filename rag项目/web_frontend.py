"""
@file : web_frontend.py
职位分析 & 技能匹配系统前端
启动：python web_frontend.py
"""
import requests
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BACKEND = "http://localhost:8000"

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════
css = """
body, .gradio-container { font-family: 'Inter', 'PingFang SC', sans-serif; }

#header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 20px;
    text-align: center;
}
#header h1 { color:#fff !important; font-size:2rem !important; font-weight:700 !important; margin:0 0 8px !important; }
#header p  { color:#a8b4c8 !important; font-size:1rem !important; margin:0 !important; }

#status-row { background:#f8fafc; border-radius:10px; padding:8px 16px; margin-bottom:16px; border:1px solid #e2e8f0; }

#search-box textarea {
    font-size:1rem !important; border-radius:10px !important;
    border:2px solid #e2e8f0 !important; padding:12px 16px !important; transition:border-color 0.2s;
}
#search-box textarea:focus { border-color:#4f8ef7 !important; box-shadow:0 0 0 3px rgba(79,142,247,0.15) !important; }

#search-btn {
    background:linear-gradient(135deg,#4f8ef7,#2563eb) !important;
    border:none !important; border-radius:10px !important; color:white !important;
    font-size:1rem !important; font-weight:600 !important; height:50px !important;
    transition:transform 0.1s, box-shadow 0.2s !important;
}
#search-btn:hover { transform:translateY(-1px) !important; box-shadow:0 4px 12px rgba(79,142,247,0.4) !important; }

/* 左侧职位列表滚动区 */
#jobs-panel {
    height: 820px;
    overflow-y: auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
}
#jobs-panel::-webkit-scrollbar { width:6px; }
#jobs-panel::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }

/* 右侧看板 */
#dashboard {
    height: 820px;
    overflow-y: auto;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
#dashboard::-webkit-scrollbar { width:6px; }
#dashboard::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }

#stats-panel {
    background:#ffffff;
    border:1px solid #e2e8f0;
    border-radius:10px;
    padding:14px 18px;
}

.gr-plot { border-radius:10px !important; border:1px solid #e2e8f0 !important; background:#fff !important; }

.tab-nav button { font-size:0.95rem !important; font-weight:500 !important; border-radius:8px 8px 0 0 !important; padding:10px 20px !important; }
.tab-nav button.selected { color:#2563eb !important; border-bottom:2px solid #2563eb !important; }
#check-btn { border-radius:8px !important; border:1px solid #e2e8f0 !important; font-size:0.85rem !important; }
"""

# ═══════════════════════════════════════════════════════
# 图表
# ═══════════════════════════════════════════════════════
CHART_BG     = '#ffffff'
CHART_COLORS = ['#3b82f6','#6366f1','#8b5cf6','#a78bfa','#c4b5fd']


def create_skills_chart(top_skills: list):
    if not top_skills:
        return None
    skills = [s[0][:25] for s in top_skills]
    counts = [s[1] for s in top_skills]

    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    bars = ax.barh(skills[::-1], counts[::-1], color=CHART_COLORS[:len(skills)], height=0.5)
    ax.bar_label(bars, padding=5, fontsize=10, color='#374151')
    ax.set_xlabel('Frequency', fontsize=10, color='#6b7280')
    ax.set_title('Top 5 Required Skills', fontsize=12, fontweight='bold', color='#111827', pad=10)
    ax.tick_params(colors='#6b7280', labelsize=9)
    for spine in ['top','right','left']: ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.set_xlim(0, max(counts) * 1.2)
    plt.tight_layout()
    return fig


def create_location_chart(locations: dict):
    if not locations:
        return None
    clean = {k: v for k, v in locations.items() if k and str(k) != 'nan'}
    if not clean:
        return None
    labels = list(clean.keys())[:6]
    sizes  = list(clean.values())[:6]
    colors = ['#3b82f6','#6366f1','#8b5cf6','#ec4899','#f59e0b','#10b981']

    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        pctdistance=0.78, colors=colors[:len(labels)],
        wedgeprops=dict(linewidth=2, edgecolor='white')
    )
    for t in texts:     t.set_fontsize(9);  t.set_color('#374151')
    for t in autotexts: t.set_fontsize(8);  t.set_color('white'); t.set_fontweight('bold')
    ax.set_title('Location Distribution', fontsize=12, fontweight='bold', color='#111827', pad=10)
    plt.tight_layout()
    return fig


def create_placeholder_chart(title: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor='#f8fafc')
    ax.set_facecolor('#f8fafc')
    ax.text(0.5, 0.55, '📊', fontsize=40, ha='center', va='center', transform=ax.transAxes, alpha=0.25)
    ax.text(0.5, 0.32, 'Search to load data', fontsize=11, ha='center', va='center',
            transform=ax.transAxes, color='#9ca3af')
    ax.set_title(title, fontsize=12, fontweight='bold', color='#d1d5db', pad=10)
    ax.axis('off')
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════
# API 调用
# ═══════════════════════════════════════════════════════

def check_backend():
    try:
        resp = requests.get(f"{BACKEND}/api/health", timeout=5)
        data = resp.json()
        return (f"🟢  Backend online  |  Job data: {data.get('job_count', 0):,}  |  "
                f"Vector index: {data.get('vector_count', 0):,}")
    except:
        return "🔴  Backend offline — please run web_backend.py first"


def _build_output(jobs: list, stats: dict, label: str = ""):
    total = stats.get('total_count', len(jobs))

    # ── 左列：职位列表 ──
    jobs_md = (
        f"<div style='margin-bottom:12px;padding:10px 14px;background:#eff6ff;"
        f"border-radius:8px;border-left:4px solid #3b82f6'>"
        f"<b style='color:#1d4ed8'>{label} — {total:,} positions</b>"
        f"&nbsp;&nbsp;<span style='color:#6b7280;font-size:0.85rem'>showing {len(jobs)}</span>"
        f"</div>\n\n"
    )
    for i, job in enumerate(jobs, 1):
        skills = job.get('skills', '')
        skill_tags = ''
        if skills:
            tag_list   = [s.strip() for s in str(skills).split(',')][:4]
            skill_tags = ' '.join(f"`{t}`" for t in tag_list if t)

        jobs_md += f"### {i}. {job.get('job_title', '-')}\n"
        jobs_md += f"🏢 **{job.get('company', '-')}**\n\n"
        jobs_md += (
            f"📍 {job.get('location', '-')}, {job.get('country', '')}　"
            f"💰 **{job.get('salary', 'N/A')}**　"
            f"⏱ {job.get('work_type', '-')}　"
            f"🎯 {job.get('experience', '-')}\n\n"
        )
        if skill_tags:
            jobs_md += f"🛠 {skill_tags}\n\n"
        desc = job.get('description', '')
        if desc:
            jobs_md += (
                f"<details><summary style='cursor:pointer;color:#6b7280;font-size:0.85rem'>"
                f"View description</summary>"
                f"<p style='color:#374151;font-size:0.9rem;margin-top:6px'>"
                f"{str(desc)[:250]}...</p></details>\n\n"
            )
        jobs_md += "<hr style='border:none;border-top:1px solid #f1f5f9;margin:12px 0'/>\n\n"

    # ── 右列：统计摘要 ──
    work_types    = stats.get("work_types", {})
    salary_ranges = stats.get("salary_ranges", {})
    top_skills    = stats.get("top_skills", [])

    top_type  = max(work_types,    key=work_types.get)    if work_types    else "—"
    top_range = max(salary_ranges, key=salary_ranges.get) if salary_ranges else "—"
    top_skill = top_skills[0][0]                          if top_skills    else "—"

    stats_md = "### 📊 Market Overview\n\n"
    stats_md += (
        f"| Metric | Value |\n|:---|:---|\n"
        f"| Total Positions | **{total:,}** |\n"
        f"| Most Common Type | **{top_type}** |\n"
        f"| Most Common Salary | **{top_range}** |\n"
        f"| Hottest Skill | **{top_skill}** |\n"
    )

    skills_fig   = create_skills_chart(top_skills)
    location_fig = create_location_chart(stats.get("locations", {}))

    return jobs_md, stats_md, skills_fig, location_fig


def load_random_jobs():
    try:
        resp = requests.get(f"{BACKEND}/api/random-jobs?n=10", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return _build_output(data.get("jobs", []), data.get("stats", {}), label="🎲 Random sample")
    except Exception:
        return (
            "<p style='color:#9ca3af;text-align:center;padding:40px'>Backend not connected</p>",
            "",
            create_placeholder_chart("Top 5 Skills"),
            create_placeholder_chart("Location Distribution"),
        )


def api_skill_match(current_major: str, current_skills: str, target_job: str):
    if not target_job.strip():
        return "⚠️ Please enter a target job", ""
    try:
        resp = requests.post(
            f"{BACKEND}/api/skill-match",
            json={"current_major": current_major,
                  "current_skills": current_skills,
                  "target_job": target_job},
            timeout=90,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})

        # ── 左列：匹配分析 ──
        score = result.get("match_score", 0)
        filled = round(score / 10)
        bar    = "█" * filled + "░" * (10 - filled)

        if score >= 70:   score_color = "#16a34a"
        elif score >= 40: score_color = "#d97706"
        else:             score_color = "#dc2626"

        left_md  = f"## 🎯 Skill Match Analysis\n\n"
        left_md += (
            f"<div style='background:#f8fafc;border-radius:10px;padding:16px 20px;"
            f"border:1px solid #e2e8f0;margin-bottom:16px'>"
            f"<div style='font-size:0.9rem;color:#6b7280;margin-bottom:6px'>Match Score</div>"
            f"<div style='font-size:2rem;font-weight:700;color:{score_color}'>{score} / 100</div>"
            f"<div style='font-family:monospace;font-size:1.1rem;color:{score_color};margin:6px 0'>{bar}</div>"
            f"<div style='color:#374151;font-size:0.95rem'>{result.get('match_analysis','')}</div>"
            f"</div>\n\n"
        )

        matched = result.get("matched_skills", [])
        if matched:
            left_md += "### ✅ Skills You Already Have\n"
            left_md += " ".join(f"`{s}`" for s in matched) + "\n\n"

        gaps = result.get("skill_gaps", [])
        if gaps:
            left_md += "### ❌ Skill Gaps\n"
            left_md += " ".join(f"`{s}`" for s in gaps) + "\n\n"

        advice = result.get("advice", "")
        if advice:
            left_md += (
                f"### 💡 Career Advice\n"
                f"<div style='background:#eff6ff;border-left:4px solid #3b82f6;"
                f"border-radius:0 8px 8px 0;padding:12px 16px;color:#1e40af'>"
                f"{advice}</div>\n\n"
            )

        # ── 右列：学习路线 ──
        right_md = "## 📚 Learning Roadmap\n\n"
        phase_colors = ["#3b82f6", "#8b5cf6", "#10b981"]

        for i, phase in enumerate(result.get("learning_roadmap", [])):
            color = phase_colors[i % len(phase_colors)]
            right_md += (
                f"<div style='border-left:4px solid {color};padding:0 0 0 16px;margin-bottom:20px'>\n\n"
                f"### {phase.get('phase','')}　<span style='font-size:0.85rem;color:#6b7280'>"
                f"{phase.get('duration','')}</span>\n\n"
                f"**Goal:** {phase.get('goal','')}\n\n"
            )
            for skill in phase.get("skills", []):
                burl = skill.get("bilibili_url", "")
                bkw  = skill.get("bilibili_keyword", skill.get("name",""))
                right_md += f"#### 🔹 {skill.get('name','')}\n"
                right_md += f"{skill.get('description','')}\n\n"
                if burl:
                    right_md += (
                        f"<a href='{burl}' target='_blank' style='display:inline-flex;"
                        f"align-items:center;gap:6px;background:#fb7299;color:white;"
                        f"padding:4px 12px;border-radius:6px;font-size:0.85rem;"
                        f"text-decoration:none;font-weight:500'>"
                        f"📺 B站搜索：{bkw}</a>\n\n"
                    )
            right_md += "</div>\n\n---\n\n"

        return left_md, right_md

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend", ""
    except Exception as e:
        return f"❌ Request failed: {e}", ""


def api_search_jobs(job_title: str):
    if not job_title.strip():
        return "⚠️ Please enter a job title", "", None, None
    try:
        resp = requests.post(f"{BACKEND}/api/search-jobs", json={"job_title": job_title}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return _build_output(data.get("jobs", []), data.get("stats", {}), label="🔍 Search results")
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend", "", None, None
    except Exception as e:
        return f"❌ Request failed: {e}", "", None, None


# ═══════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════

with gr.Blocks(title="Job Analysis & Skill Match") as demo:

    # Banner
    gr.HTML("""
    <div id="header">
        <h1>💼 Job Analysis & Skill Match</h1>
        <p>Powered by BERT · FAISS · BM25 · GPT-4o-mini</p>
    </div>
    """)

    # 状态栏
    with gr.Row(elem_id="status-row"):
        status_box = gr.Textbox(label="", interactive=False, scale=5, container=False)
        check_btn  = gr.Button("🔄 Refresh", scale=1, elem_id="check-btn", size="sm")
    check_btn.click(fn=check_backend, outputs=status_box)
    demo.load(fn=check_backend, outputs=status_box)

    # ── Tab 1：职位分析 ──────────────────────────────
    with gr.Tab("📊 Job Analysis"):

        # 搜索栏
        with gr.Row():
            job_input  = gr.Textbox(
                label="", scale=5, elem_id="search-box", container=False,
                placeholder="🔍  Enter job title, e.g.  Data Analyst · Product Manager · ML Engineer",
            )
            search_btn = gr.Button("Search", variant="primary", scale=1, elem_id="search-btn")

        gr.Examples(
            examples=["Data Analyst", "Product Manager", "Machine Learning Engineer",
                      "Software Engineer", "Marketing Manager"],
            inputs=job_input, label="Quick examples",
        )

        gr.HTML("<div style='margin:10px 0'></div>")

        # 主内容：两列等宽
        with gr.Row(equal_height=True):

            # 左列：职位列表
            with gr.Column(scale=5):
                gr.HTML("<p style='color:#6b7280;font-size:0.85rem;margin:0 0 6px 4px'>📋 Job Listings</p>")
                jobs_output = gr.Markdown(elem_id="jobs-panel")

            # 右列：分析看板
            with gr.Column(scale=5):
                gr.HTML("<p style='color:#6b7280;font-size:0.85rem;margin:0 0 6px 4px'>📊 Analytics Dashboard</p>")
                with gr.Column(elem_id="dashboard"):
                    stats_output   = gr.Markdown(elem_id="stats-panel")
                    skills_chart   = gr.Plot(label="Top 5 Required Skills")
                    location_chart = gr.Plot(label="Location Distribution")

        OUTPUTS = [jobs_output, stats_output, skills_chart, location_chart]

        search_btn.click(fn=api_search_jobs, inputs=[job_input], outputs=OUTPUTS)
        job_input.submit(fn=api_search_jobs, inputs=[job_input], outputs=OUTPUTS)
        demo.load(fn=load_random_jobs, outputs=OUTPUTS)

    # ── Tab 2：技能匹配 & 学习路线 ───────────────────
    with gr.Tab("🗺️ Skill Match & Learning Path"):

        gr.HTML("<div style='margin:10px 0'></div>")

        # 输入区
        with gr.Row():
            with gr.Column(scale=1):
                major_input = gr.Textbox(
                    label="📚 Current Major / Background",
                    placeholder="e.g. Computer Science, Information Management, Marketing",
                )
            with gr.Column(scale=1):
                skills_input = gr.Textbox(
                    label="🛠 Current Skills",
                    placeholder="e.g. Python, Excel, SQL, Data Analysis",
                    lines=2,
                )
            with gr.Column(scale=1):
                target_input = gr.Textbox(
                    label="🎯 Target Job",
                    placeholder="e.g. Data Scientist, Product Manager",
                )

        match_btn = gr.Button("🚀 Analyze Match & Generate Roadmap", variant="primary")

        gr.Examples(
            examples=[
                ["Information Management", "Excel, SQL, Python basics", "Data Analyst"],
                ["Computer Science",       "Python, ML fundamentals",  "Machine Learning Engineer"],
                ["Marketing",              "User research, Axure",     "Product Manager"],
            ],
            inputs=[major_input, skills_input, target_input],
            label="Quick examples",
        )

        gr.HTML("<div style='margin:10px 0'></div>")

        # 输出区：两列
        with gr.Row():
            with gr.Column(scale=5):
                match_output = gr.Markdown(elem_id="jobs-panel")
            with gr.Column(scale=5):
                roadmap_output = gr.Markdown(elem_id="jobs-panel")

        match_btn.click(
            fn=api_skill_match,
            inputs=[major_input, skills_input, target_input],
            outputs=[match_output, roadmap_output],
        )

demo.launch(inbrowser=True, server_port=7860, theme=gr.themes.Soft(), css=css)
