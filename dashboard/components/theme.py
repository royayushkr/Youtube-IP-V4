from __future__ import annotations

import streamlit as st


APP_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --app-canvas: #090B14;
    --app-bg: #0F1324;
    --app-bg-alt: #141A31;
    --app-surface-1: #141A31;
    --app-surface-2: #1A2140;
    --app-surface-3: #202850;
    --app-surface-soft: rgba(255, 255, 255, 0.035);
    --app-border: rgba(255, 255, 255, 0.08);
    --app-border-strong: rgba(196, 181, 253, 0.22);
    --app-text: #F7F8FC;
    --app-text-secondary: #B8C1DA;
    --app-text-tertiary: #8993B2;
    --app-accent: #8B5CF6;
    --app-accent-2: #A855F7;
    --app-accent-soft: #C4B5FD;
    --app-success: #34D399;
    --app-warning: #FBBF24;
    --app-shadow: 0 18px 42px rgba(3, 6, 20, 0.52);
    --app-shadow-glow: 0 0 0 1px rgba(139, 92, 246, 0.12), 0 24px 56px rgba(83, 44, 184, 0.24);
    --app-shadow-soft: 0 10px 24px rgba(3, 6, 20, 0.28);
    --app-radius-lg: 24px;
    --app-radius-md: 18px;
    --app-radius-pill: 999px;
    --app-control-height: 54px;
    --app-page-width: 1240px;
    --app-command-width: 1040px;
    --app-section-width: 1160px;
    --app-gap-sm: 12px;
    --app-gap-md: 16px;
    --app-gap-lg: 32px;
    --app-section-pad: 20px;

    /* Backwards-compatible aliases for existing page-level CSS */
    --yt-red: #8B5CF6;
    --yt-red-dark: #6D28D9;
    --yt-bg: #0F1324;
    --yt-bg-alt: #141A31;
    --yt-surface: #1A2140;
    --yt-accent: #A855F7;
    --yt-success: #34D399;
    --yt-warning: #FBBF24;
    --yt-text: #F7F8FC;
    --yt-text-muted: #B8C1DA;
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top center, rgba(139, 92, 246, 0.16) 0%, transparent 26%),
        radial-gradient(circle at top left, rgba(168, 85, 247, 0.14) 0%, transparent 22%),
        radial-gradient(circle at right center, rgba(96, 165, 250, 0.08) 0%, transparent 20%),
        linear-gradient(180deg, #0E1120 0%, #090B14 45%, #070912 100%) !important;
    color: var(--app-text);
    font-family: "Plus Jakarta Sans", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(9, 11, 20, 0.76) !important;
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top left, rgba(139, 92, 246, 0.18) 0%, transparent 28%),
        linear-gradient(180deg, #0F1324 0%, #141A31 48%, #0F1324 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="collapsedControl"] button {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 999px !important;
    color: #FFFFFF !important;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28) !important;
}

.block-container {
    max-width: var(--app-page-width) !important;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--app-radius-lg) !important;
    border: 1px solid var(--app-border) !important;
    background:
        radial-gradient(circle at top left, rgba(139, 92, 246, 0.12) 0%, transparent 28%),
        linear-gradient(180deg, rgba(26, 33, 64, 0.94) 0%, rgba(15, 19, 36, 0.98) 100%) !important;
    box-shadow: var(--app-shadow) !important;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.16);
    border-radius: 4px;
}

.yt-page-title {
    font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: var(--app-text);
    letter-spacing: -0.03em;
}

.yt-page-subtitle {
    color: var(--app-text-secondary);
    font-size: 15px;
    margin-bottom: 1.4rem;
    max-width: 720px;
}

.yt-section-header {
    font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
    font-size: 24px;
    font-weight: 700;
    margin-top: 1.7rem;
    margin-bottom: 0.4rem;
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    color: var(--app-text);
    letter-spacing: -0.02em;
}

.yt-section-underline {
    width: 88px;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--app-accent), var(--app-accent-soft));
    margin-bottom: 1.15rem;
    box-shadow: 0 0 18px rgba(139, 92, 246, 0.32);
}

.metric-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.metric-card {
    flex: 1 1 160px;
    padding: 0.95rem 1rem;
    border-radius: var(--app-radius-md);
    background:
        linear-gradient(180deg, rgba(36, 44, 80, 0.92) 0%, rgba(20, 26, 49, 0.98) 100%);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow);
    backdrop-filter: blur(12px);
    transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, border-color 0.15s;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--app-border-strong);
    box-shadow: var(--app-shadow-glow);
}

.metric-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--app-text-tertiary);
    margin-bottom: 0.2rem;
}

.metric-value {
    font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
    font-size: 30px;
    font-weight: 700;
    line-height: 1.08;
    color: var(--app-text);
}

.metric-icon {
    font-size: 18px;
    margin-right: 0.5rem;
}

.metric-delta {
    font-size: 12px;
    margin-top: 0.2rem;
}

.metric-delta.positive {
    color: var(--app-success);
}

.metric-delta.negative {
    color: #F472B6;
}

.styled-dataframe thead tr th {
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.92), rgba(26, 33, 64, 0.98));
    color: #FFFFFF !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.styled-dataframe tbody tr:nth-child(odd) {
    background-color: rgba(255, 255, 255, 0.015);
}

.styled-dataframe tbody tr:nth-child(even) {
    background-color: rgba(255, 255, 255, 0.028);
}

.stButton > button,
.stFormSubmitButton > button {
    min-height: var(--app-control-height) !important;
    border-radius: 18px !important;
    padding: 0.75rem 1.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    transition: transform 0.12s ease-out, box-shadow 0.12s ease-out, filter 0.15s, border-color 0.15s;
}

button[kind="primary"],
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--app-accent), var(--app-accent-2)) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(196, 181, 253, 0.18) !important;
    box-shadow: 0 14px 30px rgba(64, 24, 154, 0.34) !important;
}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 34px rgba(64, 24, 154, 0.42) !important;
    filter: brightness(1.05);
}

.stButton > button:not([kind="primary"]),
.stFormSubmitButton > button:not([kind="primary"]),
button[kind="secondary"],
button[kind="secondaryFormSubmit"] {
    background: rgba(255, 255, 255, 0.03) !important;
    color: var(--app-text) !important;
    border: 1px solid rgba(196, 181, 253, 0.16) !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), var(--app-shadow-soft) !important;
}

.stButton > button:not([kind="primary"]):hover,
.stFormSubmitButton > button:not([kind="primary"]):hover,
button[kind="secondary"]:hover,
button[kind="secondaryFormSubmit"]:hover {
    transform: translateY(-1px);
    border-color: rgba(196, 181, 253, 0.28) !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.stButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible {
    outline: none !important;
    box-shadow:
        0 0 0 3px rgba(139, 92, 246, 0.18),
        0 14px 30px rgba(54, 20, 122, 0.28) !important;
}

.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div,
.stDateInput > div > div,
.stSlider > div > div,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: rgba(15, 19, 36, 0.96) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: var(--app-text) !important;
    box-shadow: none !important;
    min-height: var(--app-control-height) !important;
}

.stTextInput > div > div,
.stNumberInput > div > div,
[data-baseweb="input"],
[data-baseweb="select"] {
    border-radius: 16px !important;
}

.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox > div > div,
.stDateInput > div > div,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    transition: border-color 0.14s ease-out, box-shadow 0.14s ease-out, background-color 0.14s ease-out !important;
}

.stTextInput > div > div:hover,
.stNumberInput > div > div:hover,
.stSelectbox > div > div:hover,
.stDateInput > div > div:hover,
[data-baseweb="select"] > div:hover,
[data-baseweb="input"] > div:hover {
    border-color: rgba(196, 181, 253, 0.20) !important;
    background-color: rgba(15, 19, 36, 0.98) !important;
}

.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    outline: none !important;
    border-color: rgba(139, 92, 246, 0.62) !important;
    box-shadow: none !important;
}

.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stSelectbox > div > div:focus-within,
.stDateInput > div > div:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
    border-color: rgba(139, 92, 246, 0.62) !important;
    box-shadow:
        0 0 0 2px rgba(139, 92, 246, 0.14),
        0 10px 24px rgba(8, 10, 26, 0.18) !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #6F7B9F !important;
    opacity: 1 !important;
}

[data-testid="stSegmentedControl"] {
    width: 100%;
}

[data-testid="stSegmentedControl"] [data-baseweb="button-group"],
[data-testid="stSegmentedControl"] [role="radiogroup"] {
    width: 100%;
    display: flex !important;
}

[data-testid="stSegmentedControl"] [data-baseweb="button-group"] > *,
[data-testid="stSegmentedControl"] [role="radiogroup"] > * {
    flex: 1 1 0 !important;
}

[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="radio"] {
    min-height: var(--app-control-height) !important;
    width: 100% !important;
    justify-content: center !important;
}

.stToggle label, .stCheckbox label, .stRadio label, .stSelectbox label,
.stDateInput label, .stTextInput label, .stSlider label, .stNumberInput label {
    color: var(--app-text-secondary) !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}

.stTabs [data-baseweb="tab"] {
    padding: 0.35rem 0 0.6rem;
}

.stTabs [data-baseweb="tab"] p {
    font-size: 14px;
    font-weight: 600;
    color: var(--app-text-secondary);
}

[aria-selected="true"] p {
    color: var(--app-text) !important;
}

.yt-card {
    border-radius: var(--app-radius-md);
    padding: 1.1rem 1.25rem;
    background:
        linear-gradient(180deg, rgba(33, 40, 73, 0.96) 0%, rgba(20, 26, 49, 0.98) 100%);
    border: 1px solid var(--app-border);
    box-shadow: var(--app-shadow);
    backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
}

.app-section-shell {
    max-width: var(--app-section-width);
    margin: 0 auto var(--app-gap-lg);
}

.app-command-shell {
    max-width: var(--app-command-width);
    margin: 0 auto var(--app-gap-lg);
}

.app-section-intro {
    margin-bottom: var(--app-gap-md);
}

.app-section-title {
    font-family: "Space Grotesk", "Plus Jakarta Sans", system-ui, sans-serif;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.06;
    letter-spacing: -0.03em;
    color: var(--app-text);
    margin-bottom: 0.22rem;
}

.app-section-copy {
    color: var(--app-text-secondary);
    font-size: 13px;
    line-height: 1.55;
    max-width: 760px;
}

.app-subsection-label {
    font-size: 11px;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--app-text-tertiary);
    margin: 0.2rem 0 0.7rem;
}

.app-meta-stack {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    margin-bottom: 1rem;
}

.app-meta-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    justify-content: flex-start;
}

.app-meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.36rem;
    padding: 0.44rem 0.72rem;
    border-radius: var(--app-radius-pill);
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--app-text-secondary);
    font-size: 12px;
    line-height: 1;
    white-space: nowrap;
}

.app-meta-pill strong {
    color: var(--app-text);
    font-weight: 700;
}

.app-meta-pill--state {
    background: rgba(139, 92, 246, 0.12);
    border-color: rgba(196, 181, 253, 0.18);
}

.keyword-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.18rem 0.62rem;
    border-radius: 999px;
    margin: 0.12rem;
    font-size: 12px;
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.22), rgba(196, 181, 253, 0.16));
    border: 1px solid rgba(196, 181, 253, 0.16);
    color: var(--app-text);
    white-space: nowrap;
}

.metric-icon {
    display: none !important;
}

.fade-in {
    animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""


def inject_shared_theme() -> None:
    st.markdown(APP_THEME_CSS, unsafe_allow_html=True)
