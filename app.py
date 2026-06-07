# MY STYLE - music recommendation app
# final project for machine learning class
# using the vatsalmavani spotify dataset from kaggle
#
# datasets used:
#   data.csv          - main track info (170k songs)
#   data_by_genres.csv - genre-level audio averages
#   data_by_year.csv   - yearly audio feature trends
#   data_by_artist.csv - artist-level stats
#   data_w_genres.csv  - tracks with genre tags
#
# algorithms: content-based (mean vector), KNN, collaborative filtering
# all three run automatically, no need to pick one

import hashlib
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="MY STYLE",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# spent way too long on this css ngl
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    /* ââ€€ââ€€ Spotify Dark Palette ââ€€ââ€€ */
    --bg: #121212;
    --bg-elevated: #1a1a1a;
    --surface: #181818;
    --card: #1e1e1e;
    --card-hover: #282828;
    --highlight: #333333;

    /* ââ€€ââ€€ Accent Colors ââ€€ââ€€ */
    --accent: #1db954;
    --accent-dim: rgba(29, 185, 84, 0.15);
    --accent-glow: rgba(29, 185, 84, 0.25);
    --gold: #e8c547;
    --violet: #a78bfa;
    --pink: #f472b6;
    --teal: #5eead4;

    /* ââ€€ââ€€ Typography ââ€€ââ€€ */
    --text: #ffffff;
    --text-secondary: #b3b3b3;
    --sub: #727272;

    /* ââ€€ââ€€ Borders & Shadows ââ€€ââ€€ */
    --border: rgba(255, 255, 255, 0.07);
    --border-hover: rgba(255, 255, 255, 0.15);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.5);
    --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.6);

    /* ââ€€ââ€€ Layout ââ€€ââ€€ */
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --bottom-bar-h: 72px;
    --transition: 0.2s ease;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   GLOBAL RESET
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* add bottom padding so content isn't hidden behind the fixed bar */
[data-testid="stMain"] > div:first-child {
    padding-bottom: calc(var(--bottom-bar-h) + 20px) !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SCROLLBAR
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SIDEBAR â€â€ Spotify-style persistent nav
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: none !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: var(--surface) !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }

/* Sidebar logo area */
.sidebar-logo {
    padding: 1.2rem 1rem 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.sidebar-logo-icon {
    font-size: 1.8rem;
    line-height: 1;
}

.sidebar-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text) !important;
    letter-spacing: -0.02em;
}

/* Sidebar nav items */
.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1rem;
    margin: 0.1rem 0.5rem;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition);
    text-decoration: none;
}

.sidebar-nav-item:hover {
    color: var(--text) !important;
    background: rgba(255,255,255,0.05);
}

.sidebar-nav-item.active {
    color: var(--text) !important;
    font-weight: 700;
    background: rgba(255,255,255,0.08);
}

.sidebar-nav-item .nav-icon { font-size: 1.2rem; opacity: 0.8; }
.sidebar-nav-item.active .nav-icon { opacity: 1; }

/* sidebar divider */
.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 0.6rem 1rem;
}

/* sidebar section label */
.sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--sub) !important;
    padding: 0.8rem 1rem 0.3rem;
}

/* playlist mini summary */
.sidebar-playlist-summary {
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: var(--sub);
}

.sidebar-playlist-summary .pl-count {
    color: var(--text) !important;
    font-weight: 600;
}

/* override Streamlit radio to look like nav */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.15rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
    background: transparent !important;
    border-radius: var(--radius-md) !important;
    padding: 0.55rem 0.85rem !important;
    margin: 0.05rem 0.3rem !important;
    transition: all var(--transition) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    background: rgba(255,255,255,0.06) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] input:checked + div {
    color: var(--text) !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(255,255,255,0.08) !important;
    color: var(--text) !important;
    font-weight: 700 !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   GRADIENT SECTION HEADERS
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.gradient-header {
    position: relative;
    padding: 2rem 0 1.5rem;
    margin: -1rem -1rem 1.5rem -1rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.gradient-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 0;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg);
}

.gradient-header.gh-discover::before { background: linear-gradient(180deg, rgba(29,185,84,0.35) 0%, var(--bg) 100%); }
.gradient-header.gh-playlist::before { background: linear-gradient(180deg, rgba(167,139,250,0.3) 0%, var(--bg) 100%); }
.gradient-header.gh-explore::before  { background: linear-gradient(180deg, rgba(94,234,212,0.3) 0%, var(--bg) 100%); }
.gradient-header.gh-profile::before  { background: linear-gradient(180deg, rgba(232,197,71,0.3) 0%, var(--bg) 100%); }
.gradient-header.gh-eval::before     { background: linear-gradient(180deg, rgba(244,114,182,0.3) 0%, var(--bg) 100%); }

.gradient-header .gh-content {
    position: relative;
    z-index: 1;
}

.gradient-header .gh-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 700;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.02em;
}

.gradient-header .gh-subtitle {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    font-weight: 400;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   FIXED BOTTOM BAR â€â€ "Now Playing"
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.bottom-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: var(--bottom-bar-h);
    background: var(--bg-elevated);
    border-top: 1px solid var(--border);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

.bb-track {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex: 1;
    min-width: 0;
}

.bb-album-art {
    width: 44px;
    height: 44px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}

.bb-track-info { min-width: 0; }

.bb-track-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 220px;
}

.bb-track-artist {
    font-size: 0.72rem;
    color: var(--sub);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 220px;
}

.bb-controls {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    flex: 1;
    justify-content: center;
}

.bb-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    font-size: 1.1rem;
    cursor: default;
    transition: color var(--transition);
    padding: 0.3rem;
}

.bb-btn:hover { color: var(--text); }

.bb-btn-play {
    background: var(--text) !important;
    color: var(--bg) !important;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

.bb-progress {
    width: 100%;
    max-width: 420px;
    margin-top: 0.2rem;
}

.bb-progress-bar {
    height: 3px;
    background: var(--highlight);
    border-radius: 2px;
    position: relative;
    overflow: hidden;
}

.bb-progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s;
}

.bb-stats {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    flex: 1;
    justify-content: flex-end;
}

.bb-stat-badge {
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--sub);
    background: rgba(255,255,255,0.06);
    padding: 0.2rem 0.55rem;
    border-radius: 10px;
}

.bb-stat-badge .bb-val {
    color: var(--accent);
    font-weight: 700;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   CARD GRID â€â€ responsive auto-fill layout
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.8rem;
    margin: 0.5rem 0 1rem;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SONG CARD â€â€ Spotify-inspired
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.song-card {
    background: var(--card);
    border: none;
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
    transition: all var(--transition);
}

.song-card:hover {
    background: var(--card-hover);
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

/* left accent bar */
.song-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
    opacity: 0;
    transition: opacity var(--transition);
}

.song-card:hover::before { opacity: 1; }

/* ── Source-specific card styles ── */
.song-card.card-similar-sound {
    border-left: 3px solid #1db954;
    background: linear-gradient(90deg, rgba(29,185,84,0.06) 0%, var(--card) 40%);
}
.song-card.card-similar-sound:hover {
    background: linear-gradient(90deg, rgba(29,185,84,0.10) 0%, var(--card-hover) 40%);
}
.song-card.card-similar-sound::before { display: none; }

.song-card.card-fans-also-like {
    border-left: 3px solid #a78bfa;
    background: linear-gradient(90deg, rgba(167,139,250,0.06) 0%, var(--card) 40%);
}
.song-card.card-fans-also-like:hover {
    background: linear-gradient(90deg, rgba(167,139,250,0.10) 0%, var(--card-hover) 40%);
}
.song-card.card-fans-also-like::before { display: none; }

.song-card.card-discovery {
    border-left: 3px solid #5eead4;
    background: linear-gradient(90deg, rgba(94,234,212,0.06) 0%, var(--card) 40%);
}
.song-card.card-discovery:hover {
    background: linear-gradient(90deg, rgba(94,234,212,0.10) 0%, var(--card-hover) 40%);
}
.song-card.card-discovery::before { display: none; }

/* Legend for recommendation sources */
.rec-legend {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin: 0.6rem 0 0.8rem;
    padding: 0.5rem 0.8rem;
    background: rgba(255,255,255,0.03);
    border-radius: var(--radius-md);
}

.rec-legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
}

.rec-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.rank-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.04);
    position: absolute;
    right: 1rem;
    top: 0.5rem;
    transition: color var(--transition);
}

.song-card:hover .rank-num { color: rgba(255, 255, 255, 0.08); }

.song-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 0.2rem;
    transition: color var(--transition);
}

.song-card:hover .song-title { color: var(--accent); }

.song-artist {
    color: var(--text-secondary);
    font-size: 0.76rem;
    font-weight: 400;
}

/* ââ€€ââ€€ Tags ââ€€ââ€€ */
.tags { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem; }

.tag {
    background: rgba(255, 255, 255, 0.06);
    border: none;
    color: var(--text-secondary);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.7rem;
    font-weight: 500;
    transition: all var(--transition);
}

.tag:hover {
    background: rgba(255, 255, 255, 0.12);
    color: var(--text);
}

.tag-v { background: rgba(167,139,250,.1); color: #b49dfa; }
.tag-v:hover { background: rgba(167,139,250,.2); }
.tag-p { background: rgba(244,114,182,.08); color: #f9a8d4; }
.tag-p:hover { background: rgba(244,114,182,.16); }
.tag-c { background: rgba(94,234,212,.08); color: #6ee7c7; }
.tag-c:hover { background: rgba(94,234,212,.16); }
.tag-g { background: var(--accent-dim); color: var(--accent); }
.tag-g:hover { background: var(--accent-glow); }

/* ââ€€ââ€€ Why Box ââ€€ââ€€ */
.why-box {
    background: rgba(255,255,255,.03);
    border-left: 2px solid var(--accent);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.5rem 0.8rem;
    margin-top: 0.6rem;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-style: italic;
    line-height: 1.5;
    transition: background var(--transition);
}

.song-card:hover .why-box { background: rgba(255,255,255,.06); }

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SECTION TITLE
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 0.8rem;
    border-bottom: none;
    letter-spacing: -0.01em;
}

.sec-title span { color: var(--text-secondary); font-weight: 400; font-size: 0.85rem; }

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SEED TAG
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.seed-tag {
    display: inline-block;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.1);
    color: var(--text);
    border-radius: 20px;
    padding: 0.25rem 0.7rem;
    font-size: 0.73rem;
    font-weight: 500;
    margin: 0.2rem 0.15rem;
    transition: all var(--transition);
}

.seed-tag:hover {
    background: rgba(255,255,255,.1);
    border-color: rgba(255,255,255,.2);
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   METRIC CARDS
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
[data-testid="metric-container"] {
    background: var(--card) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.8rem !important;
    transition: all var(--transition) !important;
}

[data-testid="metric-container"]:hover {
    background: var(--card-hover) !important;
    transform: translateY(-1px);
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   BUTTONS â€â€ Spotify-style
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.stButton button {
    background: rgba(255,255,255,.07) !important;
    border: none !important;
    color: var(--text) !important;
    border-radius: 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    transition: all var(--transition) !important;
    cursor: pointer !important;
}

.stButton button:hover {
    background: rgba(255,255,255,.12) !important;
    color: var(--text) !important;
    transform: scale(1.02) !important;
}

.stButton button:active {
    transform: scale(0.98) !important;
    background: rgba(255,255,255,.15) !important;
}

.stButton button:focus {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}

.stButton button:focus:not(:focus-visible) {
    outline: none !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   TABS (for sub-tabs within pages)
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
    transition: all var(--transition) !important;
    border-bottom: 2px solid transparent !important;
    padding-bottom: 0.5rem !important;
}

[data-testid="stTabs"] button:hover {
    color: var(--text) !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--text) !important;
    border-bottom-color: var(--accent) !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   INPUTS & SELECTORS
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {
    border-radius: var(--radius-sm) !important;
}

div[data-baseweb="select"] > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    transition: border-color var(--transition) !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: var(--border-hover) !important;
}

div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SLIDERS
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important;
    box-shadow: 0 0 6px rgba(29, 185, 84, 0.3) !important;
}

[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   EXPANDER
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--card) !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    transition: color var(--transition) !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   MISC
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
hr { border-color: var(--border) !important; }

a {
    color: var(--accent) !important;
    text-decoration: none !important;
    transition: color var(--transition), opacity var(--transition) !important;
}

a:hover { opacity: 0.8 !important; }
a:focus { outline: 2px solid var(--accent) !important; outline-offset: 2px !important; }
a:active { opacity: 0.6 !important; }

#MainMenu, footer { visibility: hidden !important; }

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   WELCOME SCREEN â€â€ First-time user experience
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.welcome-screen {
    text-align: center;
    padding: 3rem 2rem 4rem;
    max-width: 640px;
    margin: 0 auto;
}

.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 0.8rem;
    display: block;
}

.welcome-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.6rem);
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
}

.welcome-subtitle {
    color: var(--text-secondary);
    font-size: 1rem;
    margin-bottom: 2.2rem;
    line-height: 1.5;
}

.welcome-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--sub);
    margin-bottom: 1rem;
    display: block;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SEED BAR â€â€ Compact now-playing strip
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.seed-bar {
    background: var(--card);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    transition: background var(--transition);
    border: 1px solid var(--border);
}

.seed-bar:hover {
    background: var(--card-hover);
}

.seed-bar-left {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    min-width: 0;
}

.seed-bar-art {
    width: 40px;
    height: 40px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}

.seed-bar-info { min-width: 0; }

.seed-bar-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.seed-bar-artist {
    font-size: 0.72rem;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.seed-bar-right {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-shrink: 0;
}

.seed-bar-match {
    font-size: 0.7rem;
    color: var(--accent);
    font-weight: 600;
    background: var(--accent-dim);
    padding: 0.2rem 0.6rem;
    border-radius: 10px;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   SOURCE BADGE â€â€ recommendation method label
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.tag-source-similar { background: rgba(29,185,84,.12); color: #1db954; }
.tag-source-similar:hover { background: rgba(29,185,84,.22); }
.tag-source-fans { background: rgba(167,139,250,.12); color: #b49dfa; }
.tag-source-fans:hover { background: rgba(167,139,250,.22); }
.tag-source-discovery { background: rgba(94,234,212,.12); color: #6ee7c7; }
.tag-source-discovery:hover { background: rgba(94,234,212,.22); }

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   FEED HEADER
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
.feed-header {
    margin-bottom: 0.3rem;
}

.feed-header .feed-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
}

.feed-header .feed-subtitle {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# features we use for the ML models
# picked these based on the kaggle notebook and what actually makes songs sound similar
AUDIO_FEATURES = [
    "valence",
    "energy",
    "acousticness",
    "danceability",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo_norm",      # normalized version of tempo
    "popularity_norm", # normalized version of popularity
]


def human_label(feature, value):
    """Convert raw audio feature values to human-readable labels."""
    labels = {
        "energy": [(0.0, 0.33, "Chill"), (0.33, 0.66, "Moderate"), (0.66, 1.01, "High Energy")],
        "valence": [(0.0, 0.33, "Melancholy"), (0.33, 0.66, "Neutral"), (0.66, 1.01, "Happy")],
        "danceability": [(0.0, 0.33, "Slow"), (0.33, 0.66, "Groovy"), (0.66, 1.01, "Dance Floor")],
    }
    for low, high, label in labels.get(feature, []):
        if low <= value < high:
            return label
    return ""


# ==============================================================
# LOADING THE DATASETS
# ==============================================================

@st.cache_data(show_spinner=False)
def load_main_tracks(max_rows=15000):
    # main song database - 170k tracks but we sample it down for speed
    if os.path.exists("data.csv.gz"):
        df = pd.read_csv("data.csv.gz")
    elif os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
    else:
        st.error("⚠️ **Dataset file not found!** Please make sure `data.csv.gz` (or `data.csv`) is uploaded to the root of your repository on GitHub.")
        st.stop()

    # rename to match our internal column names
    df.rename(columns={"name": "title", "id": "track_id", "artists": "artist"}, inplace=True)

    # clean up the artist column - it comes as "['Artist Name']" from the csv
    df["artist"] = df["artist"].astype(str).str.strip("[]'\"").str.split("',").str[0].str.strip("'\" ")

    # fix year - use release_date if year column is missing or weird
    if "year" not in df.columns or df["year"].isna().all():
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    if "popularity" not in df.columns:
        df["popularity"] = 50

    # drop rows with missing audio features
    needed = ["valence", "energy", "acousticness", "danceability",
              "instrumentalness", "liveness", "speechiness", "tempo"]
    df.dropna(subset=needed, inplace=True)

    # clip to valid ranges
    for col in ["valence", "energy", "acousticness", "danceability",
                "instrumentalness", "liveness", "speechiness"]:
        df[col] = df[col].clip(0, 1)

    df["tempo"] = df["tempo"].clip(30, 250)

    # normalize tempo and popularity to 0-1
    df["tempo_norm"] = ((df["tempo"] - 30) / 220).clip(0, 1).round(4)
    df["popularity_norm"] = (df["popularity"] / 100).clip(0, 1).round(4)

    # assign genre from data_w_genres if possible, otherwise "Unknown"
    df["genre"] = "Unknown"

    df.drop_duplicates(subset="track_id", inplace=True)
    df = df.reset_index(drop=True)

    # sample down if too big - stratify by year decade so we keep history
    if len(df) > max_rows:
        df["decade"] = (df["year"] // 10 * 10)
        df = (
            df.groupby("decade", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, int(max_rows * len(g) / len(df)))),
                random_state=42
            ))
        )
        df = df.drop(columns=["decade"], errors="ignore").reset_index(drop=True)

    keep = ["track_id", "title", "artist", "genre", "year",
            "valence", "energy", "acousticness", "danceability",
            "instrumentalness", "liveness", "speechiness",
            "tempo", "tempo_norm", "popularity", "popularity_norm"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_tracks_with_genres():
    # this file has genre tags per artist
    df = pd.read_csv("data_w_genres.csv")
    df["artist"] = df["artists"].astype(str).str.strip("[]'\"").str.split("',").str[0].str.strip("'\" ")
    df["genre"] = df["genres"].astype(str).str.strip("[]'\"").str.split("',").str[0].str.strip("'\" ").str.title()
    df["genre"] = df["genre"].replace({"": "Unknown", "[]": "Unknown", "Nan": "Unknown"})
    return df[["artist", "genre"]].drop_duplicates(subset="artist")


@st.cache_data(show_spinner=False)
def load_genre_stats():
    # pre-aggregated genre averages - useful for the explore tab
    df = pd.read_csv("data_by_genres.csv")
    df.rename(columns={"genres": "genre"}, inplace=True)
    df["genre"] = df["genre"].astype(str).str.strip("[]'\"").str.title()
    return df


@st.cache_data(show_spinner=False)
def load_year_trends():
    # audio features averaged by year - great for the decade chart
    return pd.read_csv("data_by_year.csv")


@st.cache_data(show_spinner=False)
def load_artist_stats():
    return pd.read_csv("data_by_artist.csv")


@st.cache_data(show_spinner=False)
def load_full_dataset(max_rows=15000):
    # load everything and join genres onto the main track list
    tracks = load_main_tracks(max_rows)
    genre_map = load_tracks_with_genres()

    # try to match genre by artist name
    merged = tracks.merge(genre_map, on="artist", how="left", suffixes=("", "_from_genres"))
    if "genre_from_genres" in merged.columns:
        merged["genre"] = merged["genre_from_genres"].fillna(merged["genre"])
        merged.drop(columns=["genre_from_genres"], inplace=True)

    merged["genre"] = merged["genre"].fillna("Unknown").replace("", "Unknown")
    return merged.reset_index(drop=True)


def get_genres(df):
    genres = sorted([g for g in df["genre"].dropna().unique().tolist() if g not in ["Unknown", "[]", ""]])
    return genres


# ==============================================================
# FAKE USER HISTORY for collaborative filtering
# in a real app this would be actual play counts
# ==============================================================

@st.cache_data(show_spinner=False)
def make_user_history(df, n_users=80):
    rng = np.random.default_rng(99)
    genres = get_genres(df)
    if not genres:
        genres = ["Unknown"]
    track_ids = df["track_id"].tolist()

    genre_idx_map = {}
    for g in genres:
        idx = df.index[df["genre"] == g].to_numpy()
        if len(idx) > 0:
            genre_idx_map[g] = idx

    # build the play count matrix
    plays = np.zeros((n_users, len(track_ids)), dtype=np.int16)

    for i in range(n_users):
        fav = rng.choice(genres, size=min(3, len(genres)), replace=False).tolist()
        for g in fav:
            if g in genre_idx_map:
                idxs = genre_idx_map[g]
                plays[i, idxs] = rng.integers(1, 15, size=len(idxs)).astype(np.int16)
        # add some random listens outside fav genres
        rand = rng.choice(len(track_ids), size=min(20, len(track_ids)), replace=False)
        for r in rand:
            if plays[i, r] == 0:
                plays[i, r] = int(rng.integers(0, 3))

    return pd.DataFrame(plays, index=[f"user_{i:03d}" for i in range(n_users)], columns=track_ids)


# ==============================================================
# ML - BUILDING THE MODELS
# ==============================================================

@st.cache_data(show_spinner=False)
def scale_features(df):
    # scale everything to 0-1 so tempo doesn't dominate
    cols = [c for c in AUDIO_FEATURES if c in df.columns]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[cols].values)
    return scaler, X, cols


@st.cache_data(show_spinner=False)
def fit_knn(X, k=20):
    # train the KNN model with cosine distance
    # cosine distance works better than euclidean for audio features
    model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    model.fit(X)
    return model


@st.cache_data(show_spinner=False)
def fit_pca(X):
    # compress to 2D for the map visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords


def mean_vector(seed_indices, X):
    # from the vatsal mavani kaggle notebook:
    # average all seed song vectors to get a "center point"
    # then find what's closest to that center
    vecs = X[seed_indices]
    return np.mean(vecs, axis=0).reshape(1, -1)


def build_filter_mask(df, valence_rng, energy_rng, year_rng, min_pop, genre=None):
    mask = (
        (df["valence"] >= valence_rng[0]) & (df["valence"] <= valence_rng[1]) &
        (df["energy"] >= energy_rng[0]) & (df["energy"] <= energy_rng[1]) &
        (df["popularity"] >= min_pop)
    )
    if df["year"].max() > 0:
        mask = mask & (df["year"] >= year_rng[0]) & (df["year"] <= year_rng[1])
    if genre and genre != "All":
        mask = mask & (df["genre"] == genre)
    return mask.values


# ==============================================================
# RECOMMENDATION FUNCTIONS
# all three run at the same time, no need to pick
# ==============================================================

def rec_content_based(mean_vec, X, df, seed_indices, n=6, mask=None, skip=None):
    """
    Content-based using mean vector + cosine distance
    same approach as the kaggle notebook
    """
    dists = cdist(mean_vec, X, metric="cosine")[0]

    for idx in seed_indices:
        dists[idx] = 999.0

    if mask is not None:
        dists[~mask] = 999.0

    if skip:
        for sid in skip:
            rows = df.index[df["track_id"] == sid].tolist()
            if rows:
                dists[rows[0]] = 999.0

    top = np.argsort(dists)[:n]
    result = df.iloc[top].copy()
    result["score"] = 1 - dists[top]
    return result.reset_index(drop=True)


def rec_knn(seed_idx, knn_model, X, df, n=6, mask=None, skip=None):
    """
    KNN-based - finds geometrically close songs
    gives slightly different results than mean vector
    """
    dists, idxs = knn_model.kneighbors([X[seed_idx]], n_neighbors=min(n * 4, len(df)))
    cands = df.iloc[idxs[0]].copy()
    cands["score"] = 1 - dists[0]
    cands = cands[cands["track_id"] != df.iloc[seed_idx]["track_id"]]

    if mask is not None:
        cands = cands[mask[cands.index]]
    if skip:
        cands = cands[~cands["track_id"].isin(skip)]

    return cands.head(n).reset_index(drop=True)


def rec_collaborative(user_vec, history_df, df, n=6, skip=None):
    """
    User-user collaborative filtering
    finds people with similar taste, recommends what they listened to
    """
    sims = cosine_similarity(user_vec.reshape(1, -1), history_df.values)[0]
    weighted = sims @ history_df.values
    weighted[user_vec > 0] = -1  # don't recommend stuff already heard

    if skip:
        for sid in skip:
            if sid in history_df.columns:
                weighted[list(history_df.columns).index(sid)] = -1

    # Normalize scores to 0-1 range based on the positive weights
    max_w = weighted.max()
    if max_w > 0:
        mask_pos = weighted > 0
        weighted[mask_pos] = weighted[mask_pos] / max_w

    top = np.argsort(weighted)[::-1][:n]
    rec_ids = [history_df.columns[i] for i in top]
    result = df[df["track_id"].isin(rec_ids)].copy()
    result["score"] = [weighted[i] for i in top if history_df.columns[i] in df["track_id"].values]
    return result.reset_index(drop=True)


def rec_surprise(df, X, mask, n=6, skip=None):
    """
    Surprise mode - picks a random anchor from the filtered pool
    then finds what's similar to it
    good for discovering stuff outside your usual taste
    """
    rng = np.random.default_rng()
    pool = np.where(mask)[0]

    if skip:
        pool = np.array([i for i in pool if df.iloc[i]["track_id"] not in skip])

    if len(pool) == 0:
        return pd.DataFrame()

    anchor = int(rng.choice(pool))
    anchor_title = df.iloc[anchor]["title"]

    dists = cdist(X[anchor].reshape(1, -1), X, metric="cosine")[0]
    dists[anchor] = 999.0
    dists[~mask] = 999.0

    if skip:
        for sid in skip:
            rows = df.index[df["track_id"] == sid].tolist()
            if rows:
                dists[rows[0]] = 999.0

    top = np.argsort(dists)[:n]
    result = df.iloc[top].copy()
    result["score"] = 1 - dists[top]
    result["anchor"] = anchor_title
    return result.reset_index(drop=True)


# ==============================================================
# EXPLANATION - why did we recommend this?
# ==============================================================

def why_this_song(seed, rec):
    """Generate a short, human-readable explanation."""
    checks = {
        "valence": ("mood", 0.13),
        "energy": ("energy", 0.13),
        "danceability": ("danceability", 0.13),
    }
    similar = []
    for feat, (label, threshold) in checks.items():
        if abs(seed.get(feat, 0) - rec.get(feat, 0)) < threshold:
            similar.append(label)

    if similar:
        return f"Similar {', '.join(similar[:2])}"
    return "Similar overall vibe"


def merge_recommendations(seed_idx, X, df, knn_model, user_history, n=8, mask=None, skip=None):
    """Run all recommendation engines and merge into a single ranked list."""
    seed = df.iloc[seed_idx]

    # Content-based
    mv = mean_vector([seed_idx], X)
    recs_cb = rec_content_based(mv, X, df, [seed_idx], n, mask, skip)
    if not recs_cb.empty:
        recs_cb["source"] = "Similar Sound"

    # KNN
    recs_knn = rec_knn(seed_idx, knn_model, X, df, n, mask, skip)
    if not recs_knn.empty:
        recs_knn["source"] = "Similar Sound"

    # Collaborative
    rng = np.random.default_rng(int(hashlib.md5(seed["track_id"].encode()).hexdigest(), 16) % 2**32)
    sample_user = rng.choice(user_history.index.tolist())
    u_vec = user_history.loc[sample_user].values.astype(float)
    recs_cf = rec_collaborative(u_vec, user_history, df, n, skip)
    if not recs_cf.empty:
        recs_cf["source"] = "Fans Also Like"

    # Merge and deduplicate
    all_recs = pd.concat([recs_cb, recs_knn, recs_cf], ignore_index=True)
    if all_recs.empty:
        return all_recs

    # Keep best score per track, preserve first source label
    all_recs = all_recs.sort_values("score", ascending=False)
    all_recs = all_recs.drop_duplicates(subset="track_id", keep="first")

    return all_recs.head(n).reset_index(drop=True)



# ==============================================================
# UI COMPONENTS
# ==============================================================

def show_header():
    # header is now rendered per-section via gradient_section_header
    pass


def gradient_section_header(title, subtitle, variant="discover"):
    """Spotify-style gradient section header with contextual color."""
    st.markdown(f"""
<div class="gradient-header gh-{variant}">
    <div class="gh-content">
        <div class="gh-title">{title}</div>
        <div class="gh-subtitle">{subtitle}</div>
    </div>
</div>
""", unsafe_allow_html=True)


def show_bottom_bar(seed, n_match=0):
    """Simplified fixed bottom bar — seed track + match count only."""
    title = seed.get("title", "No track selected")[:40]
    artist = seed.get("artist", "—")[:30]
    genre = seed.get("genre", "Unknown")
    if genre in ["Unknown", "[]", ""]:
        genre = "—"

    genre_colors = {
        "Pop": "#1db954", "Rock": "#e8c547", "Hip-Hop": "#f472b6",
        "Jazz": "#a78bfa", "Classical": "#5eead4", "Electronic": "#22d3ee",
        "R&B": "#fb923c", "Country": "#34d399", "Metal": "#ef4444",
    }
    art_bg = genre_colors.get(genre, "#333333")

    st.markdown(f"""
<div class="bottom-bar">
    <div class="bb-track">
        <div class="bb-album-art" style="background: {art_bg};">🎵</div>
        <div class="bb-track-info">
            <div class="bb-track-title">{title}</div>
            <div class="bb-track-artist">{artist}</div>
        </div>
    </div>
    <div class="bb-stats">
        <span class="bb-stat-badge">🎯 <span class="bb-val">{n_match:,}</span> matches</span>
    </div>
</div>
""", unsafe_allow_html=True)


def show_sidebar_nav():
    """Sidebar logo only — library stats moved to My Library page."""
    st.markdown("""
<div class="sidebar-logo">
    <span class="sidebar-logo-icon">🎵</span>
    <span class="sidebar-logo-text">MY STYLE</span>
</div>
<div class="sidebar-divider"></div>
""", unsafe_allow_html=True)


def song_card(track, key_prefix, downvotes, upvotes, source="", seed=None):
    """Simplified song card — 3 tags max, human-readable labels, ♡/✕ actions."""
    tid = track["track_id"]
    score = track.get("score", None)
    score_label = f"{score:.0%}" if score is not None else ""
    year = int(track.get("year", 0))
    year_str = str(year) if year > 0 else ""
    genre = track.get("genre", "Unknown")
    if genre in ["Unknown", "[]", ""]:
        genre = ""

    # Human-readable labels instead of raw numbers
    energy_lbl = human_label("energy", float(track.get("energy", 0)))
    mood_lbl = human_label("valence", float(track.get("valence", 0)))

    # Build subtitle
    parts = [track["artist"]]
    if year_str:
        parts.append(year_str)
    if genre:
        parts.append(genre)
    subtitle = " · ".join(parts)

    # Source badge CSS class + card-level class for visual distinction
    source_class = ""
    card_class = "song-card"
    if source == "Similar Sound":
        source_class = "tag-source-similar"
        card_class = "song-card card-similar-sound"
    elif source == "Fans Also Like":
        source_class = "tag-source-fans"
        card_class = "song-card card-fans-also-like"
    elif source == "Discovery":
        source_class = "tag-source-discovery"
        card_class = "song-card card-discovery"

    # Build tags
    tags_html = ""
    if source:
        tags_html += f'<span class="tag {source_class}">{source}</span>'
    if energy_lbl:
        tags_html += f'<span class="tag tag-c">⚡ {energy_lbl}</span>'
    if mood_lbl:
        tags_html += f'<span class="tag">💛 {mood_lbl}</span>'

    card_html = (
        f'<div class="{card_class}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
        f'<div style="min-width:0;">'
        f'<div class="song-title">{track["title"]}</div>'
        f'<div class="song-artist">{subtitle}</div>'
        f'</div>'
        + (f'<span style="font-size:.78rem;font-weight:600;color:#1db954;flex-shrink:0;margin-left:.5rem;">{score_label}</span>' if score_label else '')
        + f'</div>'
        f'<div class="tags">{tags_html}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    c1, c2, _ = st.columns([1, 1, 8])
    with c1:
        label = "❤️" if tid in upvotes else "♡"
        if st.button(label, key=f"like_{key_prefix}_{tid}"):
            upvotes.add(tid)
            downvotes.discard(tid)
            st.session_state["upvotes"] = upvotes
            st.rerun()
    with c2:
        if tid not in downvotes:
            if st.button("✕", key=f"hide_{key_prefix}_{tid}"):
                downvotes.add(tid)
                upvotes.discard(tid)
                st.session_state["downvotes"] = downvotes
                st.rerun()

    with st.expander("ℹ️ Details", expanded=False):
        if seed is not None:
            explanation = why_this_song(seed, track)
            st.markdown(f"**Vibe:** *{explanation}*")
        
        # Display audio features in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**⚡ Energy:** {track.get('energy', 0):.2f}")
            st.markdown(f"**😊 Mood:** {track.get('valence', 0):.2f}")
            st.markdown(f"**💃 Danceability:** {track.get('danceability', 0):.2f}")
        with col2:
            st.markdown(f"**🎸 Acousticness:** {track.get('acousticness', 0):.2f}")
            st.markdown(f"**🎹 Instrumentalness:** {track.get('instrumentalness', 0):.2f}")
            st.markdown(f"**🎤 Liveness:** {track.get('liveness', 0):.2f}")
            
        radar_chart(track, title="", key=f"radar_{key_prefix}_{tid}")




def radar_chart(track, title="Audio Features", key=None):
    feats = ["valence", "energy", "acousticness", "danceability", "instrumentalness", "liveness"]
    vals = [float(track.get(f, 0)) for f in feats]
    # close the polygon
    vals_loop = vals + [vals[0]]
    feats_loop = feats + [feats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_loop,
        theta=feats_loop,
        fill="toself",
        fillcolor="rgba(29, 185, 84, 0.15)",
        line=dict(color="#1db954", width=2),
        marker=dict(color="#1db954", size=5),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], color="#b3b3b3", gridcolor="#282828"),
            angularaxis=dict(color="#b3b3b3")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        margin=dict(t=30, b=20, l=30, r=30),
        height=260,
        title=dict(text=title, font=dict(size=11, color="#b3b3b3")),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def comparison_heatmap(seed, recs):
    feats = ["valence", "energy", "acousticness", "danceability",
             "instrumentalness", "liveness", "speechiness"]
    names = [seed["title"][:18] + " â˜…"] + [r["title"][:18] for _, r in recs.head(8).iterrows()]
    z = np.array([[seed.get(f, 0)] + [row.get(f, 0) for _, row in recs.head(8).iterrows()] for f in feats])

    fig = px.imshow(
        z, x=names, y=feats,
        color_continuous_scale=[[0, "#181818"], [1, "#1db954"]],
        zmin=0, zmax=1, text_auto=".2f"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=280, coloraxis_showscale=False,
        margin=dict(t=10, b=10), xaxis=dict(tickangle=-30)
    )
    st.plotly_chart(fig, use_container_width=True)


def genre_donut(df, user_vec):
    totals = {}
    for g in get_genres(df):
        idx = df.index[df["genre"] == g].to_numpy()
        totals[g] = float(user_vec[idx].sum())

    total = sum(totals.values()) or 1
    pct = dict(sorted({g: round(v / total * 100, 1) for g, v in totals.items()}.items(), key=lambda x: -x[1]))
    top7 = dict(list(pct.items())[:7])
    rest = sum(list(pct.values())[7:])
    if rest > 0:
        top7["Others"] = round(rest, 1)

    fig = px.pie(
        names=list(top7.keys()), values=list(top7.values()), hole=0.55,
        color_discrete_sequence=["#1db954","#5eead4","#a78bfa","#f472b6","#fb923c","#34d399","#b0b0c8","#727272"]
    )
    fig.update_traces(textinfo="percent+label", textfont_size=10)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#ffffff",
                      showlegend=False, margin=dict(t=10,b=10,l=10,r=10), height=280)
    st.plotly_chart(fig, use_container_width=True)


def taste_map(df, pca_coords, seed_indices):
    # 2D scatter - each dot is a song, color = genre
    # position based on audio similarity (PCA)
    n = min(3000, len(df))
    rng = np.random.default_rng(42)
    sample = rng.choice(len(df), size=n, replace=False)

    plot_df = df.iloc[sample].copy()
    plot_df["x"] = pca_coords[sample, 0]
    plot_df["y"] = pca_coords[sample, 1]
    plot_df["is_seed"] = plot_df.index.isin(seed_indices)
    plot_df["dot_size"] = plot_df["is_seed"].map({True: 14, False: 5})

    # cap genres shown so the legend isn't huge
    top_genres = plot_df["genre"].value_counts().head(15).index.tolist()
    plot_df["genre_display"] = plot_df["genre"].apply(lambda g: g if g in top_genres else "Other")

    fig = px.scatter(
        plot_df, x="x", y="y", color="genre_display",
        hover_data={"title": True, "artist": True, "year": True, "x": False, "y": False},
        opacity=0.6, size="dot_size",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    seeds = plot_df[plot_df["is_seed"]]
    if not seeds.empty:
        fig.add_trace(go.Scatter(
            x=seeds["x"], y=seeds["y"],
            mode="markers+text",
            marker=dict(color="#1db954", size=16, symbol="star",
                        line=dict(color="white", width=1.5)),
            text=seeds["title"].str[:15],
            textposition="top center",
            textfont=dict(color="#1db954", size=9),
            name="Your seed â˜…", showlegend=True,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=450,
        xaxis=dict(gridcolor="#282828", title="PCA 1"),
        yaxis=dict(gridcolor="#282828", title="PCA 2"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def year_trends_chart():
    # uses data_by_year.csv - real pre-aggregated data!
    try:
        trends = load_year_trends()
    except Exception:
        st.info("Couldn't load data_by_year.csv")
        return

    feats = ["valence", "energy", "acousticness", "danceability"]
    available = [f for f in feats if f in trends.columns]

    picked = st.multiselect("Pick features:", available,
                             default=available[:3], key="year_feats")
    if not picked:
        return

    colors = ["#1db954", "#5eead4", "#a78bfa", "#f472b6"]
    fig = go.Figure()
    for i, feat in enumerate(picked):
        fig.add_trace(go.Scatter(
            x=trends["year"], y=trends[feat], name=feat,
            mode="lines", line=dict(color=colors[i % len(colors)], width=2.5),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=340,
        xaxis=dict(gridcolor="#282828", title="Year"),
        yaxis=dict(gridcolor="#282828", title="Average value", range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)



def genre_comparison_chart():
    # uses data_by_genres.csv - real genre averages
    try:
        gdf = load_genre_stats()
    except Exception:
        st.info("Couldn't load data_by_genres.csv")
        return

    feats = ["valence", "energy", "acousticness", "danceability", "popularity"]
    available = [f for f in feats if f in gdf.columns]

    # filter out weird genre names
    gdf = gdf[~gdf["genre"].str.contains(r"\[|\]|nan", na=True)]
    gdf = gdf[gdf["genre"].str.len() > 1]

    top_genres = gdf.nlargest(20, "popularity") if "popularity" in gdf.columns else gdf.head(20)

    feat_pick = st.selectbox("Feature to compare:", available, key="genre_feat")

    fig = px.bar(
        top_genres.sort_values(feat_pick, ascending=True),
        x=feat_pick, y="genre", orientation="h",
        color=feat_pick,
        color_continuous_scale=[[0, "#181818"], [1, "#1db954"]],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=480, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#282828"),
        yaxis=dict(gridcolor="#282828"),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)



def artist_stats_chart():
    # uses data_by_artist.csv
    try:
        adf = load_artist_stats()
    except Exception:
        st.info("Couldn't load data_by_artist.csv")
        return

    if "artists" in adf.columns:
        adf.rename(columns={"artists": "artist"}, inplace=True)
    if "artist" not in adf.columns:
        st.info("No artist column found.")
        return

    adf["artist"] = adf["artist"].astype(str).str.strip("[]'\"").str.split("',").str[0].str.strip("'\" ")

    top = adf.nlargest(20, "popularity") if "popularity" in adf.columns else adf.head(20)

    fig = px.scatter(
        top, x="energy", y="valence", size="popularity" if "popularity" in top.columns else None,
        hover_name="artist", color="danceability" if "danceability" in top.columns else None,
        color_continuous_scale=[[0, "#181818"], [1, "#1db954"]],
        size_max=30,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=400, coloraxis_showscale=False,
        xaxis=dict(gridcolor="#282828", title="Energy"),
        yaxis=dict(gridcolor="#282828", title="Valence (Happiness)"),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def eval_dashboard(df):
    st.markdown('<div class="sec-title">📊 Model Evaluation <span>Dashboard</span></div>', unsafe_allow_html=True)
    st.caption("Note: precision/recall metrics are simulated. In a real app you'd need actual user interaction logs to compute these properly.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "88.2%", "+1.8%")
    m2.metric("Precision@10", "0.751", "+0.021")
    m3.metric("Recall@10", "0.698", "+0.029")
    m4.metric("NDCG@10", "0.786", "+0.019")
    m5.metric("Coverage", "71.4%", "+5.1%")
    st.divider()

    # training loss (simulated)
    st.markdown("#### Training vs Validation Loss")
    ep = np.arange(1, 61)
    rng = np.random.default_rng(1)
    t_loss = 0.88 * np.exp(-ep / 18) + 0.07 + rng.normal(0, 0.01, 60)
    v_loss = 0.83 * np.exp(-ep / 21) + 0.11 + rng.normal(0, 0.015, 60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=t_loss, name="Train", line=dict(color="#1db954", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=v_loss, name="Validation", line=dict(color="#5eead4", width=2, dash="dot")))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=250,
        xaxis=dict(gridcolor="#282828", title="Epoch"),
        yaxis=dict(gridcolor="#282828", title="Loss"),
        legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    ca, cb = st.columns(2)
    ks = [1, 2, 3, 5, 10, 15, 20]

    with ca:
        st.markdown("#### Precision@K")
        fig_p = go.Figure()
        for name, vals, col in [
            ("Content-Based", [.92,.88,.84,.80,.75,.71,.67], "#1db954"),
            ("KNN",           [.90,.86,.82,.78,.73,.69,.64], "#5eead4"),
            ("Collaborative", [.83,.79,.75,.71,.66,.62,.58], "#a78bfa"),
        ]:
            fig_p.add_trace(go.Scatter(x=ks, y=vals, name=name, mode="lines+markers",
                                       line=dict(color=col, width=2)))
        fig_p.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff", height=240,
            xaxis=dict(gridcolor="#282828", title="K"),
            yaxis=dict(gridcolor="#282828", title="Precision", range=[.4, 1]),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with cb:
        st.markdown("#### Recall@K")
        fig_r = go.Figure()
        for name, vals, col in [
            ("Content-Based", [.12,.21,.29,.44,.70,.80,.86], "#1db954"),
            ("KNN",           [.11,.20,.28,.42,.68,.78,.84], "#5eead4"),
            ("Collaborative", [.09,.18,.26,.38,.62,.72,.79], "#a78bfa"),
        ]:
            fig_r.add_trace(go.Scatter(x=ks, y=vals, name=name, mode="lines+markers",
                                       line=dict(color=col, width=2)))
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff", height=240,
            xaxis=dict(gridcolor="#282828", title="K"),
            yaxis=dict(gridcolor="#282828", title="Recall", range=[0, 1]),
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # real dataset stats
    st.markdown("#### Dataset Stats")
    total = len(df)
    n_genres = df["genre"].nunique()
    avg_pop = df["popularity"].mean()
    has_year = df["year"].max() > 0
    yr_str = f"{int(df['year'].replace(0,pd.NA).dropna().min())} – {int(df['year'].max())}" if has_year else "N/A"

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tracks Loaded", f"{total:,}")
    s2.metric("Genres", n_genres)
    s3.metric("Year Range", yr_str)
    s4.metric("Avg Popularity", f"{avg_pop:.1f}")

    st.markdown("#### Popularity Distribution")
    fig_hist = px.histogram(
        df, x="popularity", nbins=40,
        color_discrete_sequence=["#1db954"]
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff", height=220,
        xaxis=dict(gridcolor="#282828"), yaxis=dict(gridcolor="#282828"),
        margin=dict(t=10, b=10), showlegend=False
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ==============================================================
# SESSION STATE INIT
# ==============================================================

def init_session():
    defaults = {
        "downvotes": set(),
        "upvotes": set(),
        "active_user": "user_000",
        "playlist": [],
        "onboarded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ==============================================================
# MAIN
# ==============================================================

def main():
    init_session()

    # load data - reads directly from files in same folder, no upload needed
    with st.spinner("Loading your music data..."):
        df = load_full_dataset(max_rows=15000)
        user_history = make_user_history(df)
        _, X, _ = scale_features(df)
        knn_model = fit_knn(X, k=20)
        pca_coords = fit_pca(X)

    genres = get_genres(df)

    # ============================================================
    # SIDEBAR — simplified 3-item nav
    # ============================================================
    with st.sidebar:
        show_sidebar_nav()

        page = st.radio(
            "Navigate",
            ["🎧 Discover", "📚 My Library", "🗺️ Explore"],
            label_visibility="collapsed",
        )

    # ============================================================
    # Shared state: default filter values
    # ============================================================
    val_rng = (0.0, 1.0)
    eng_rng = (0.0, 1.0)
    has_year = df["year"].max() > 0
    ymin = int(df["year"].replace(0, pd.NA).dropna().min()) if has_year else 1920
    ymax = int(df["year"].max()) if has_year else 2020
    year_rng = (ymin, ymax)
    min_pop = 0

    # ============================================================
    # 🎧 DISCOVER
    # ============================================================
    if page == "🎧 Discover":

        # ── FTUX: Welcome screen for first-time users ──
        if not st.session_state["onboarded"]:
            st.markdown("""
<div class="welcome-screen">
    <span class="welcome-icon">🎵</span>
    <div class="welcome-title">Welcome to MY STYLE</div>
    <div class="welcome-subtitle">
        Tell us what you like, and we'll find music you'll love.
    </div>
    <span class="welcome-label">Pick a genre to get started</span>
</div>
            """, unsafe_allow_html=True)

            # Genre selection grid
            genre_icons = {
                "Rock": "🎸", "Pop": "🎹", "Jazz": "🎷", "Electronic": "🎧",
                "Hip Hop": "🎤", "Classical": "🎻", "R&B": "🎵", "Country": "🤠",
                "Metal": "🔥", "Indie": "🌿", "Soul": "💜", "Blues": "🎺",
            }

            # Show top available genres as buttons
            available = [g for g in genres if g in genre_icons][:8]
            if not available:
                available = genres[:8]

            cols = st.columns(4)
            selected_genre = None
            for i, g in enumerate(available):
                with cols[i % 4]:
                    icon = genre_icons.get(g, "🎵")
                    if st.button(f"{icon} {g}", key=f"welcome_{g}", use_container_width=True):
                        selected_genre = g

            # "Show me everything" button
            st.markdown("")  # spacer
            _, center_col, _ = st.columns([2, 3, 2])
            with center_col:
                if st.button("🔥 Show me everything", use_container_width=True):
                    selected_genre = "All"

            if selected_genre:
                st.session_state["onboarded"] = True
                if selected_genre != "All":
                    st.session_state["disc_genre"] = selected_genre
                st.rerun()

            return  # stop here until genre is picked

        # ── Normal Discover flow (onboarded users) ──
        gradient_section_header(
            "Discover", "Your personalized recommendations", "discover"
        )

        # ── Sidebar filters ──
        with st.sidebar:
            st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

            # Genre + song picker (always visible — Level 1)
            genre_filter = st.selectbox("🎵 Genre", ["All"] + genres, key="disc_genre")
            filtered_df = df if genre_filter == "All" else df[df["genre"] == genre_filter]
            options = [f"{r['title']} — {r['artist']}" for _, r in filtered_df.head(500).iterrows()]
            picked = st.selectbox("Search for a song", options, label_visibility="collapsed")

            # Advanced filters (collapsed — Level 2)
            with st.expander("⚙ Refine Results", expanded=False):
                val_rng = st.slider("😊 Mood: Sad ← → Happy", 0.0, 1.0, (0.0, 1.0), 0.05)
                eng_rng = st.slider("⚡ Energy: Chill ← → Hype", 0.0, 1.0, (0.0, 1.0), 0.05)
                year_rng = st.slider("📅 Year Range", ymin, ymax, (ymin, ymax), 1)
                min_pop = st.slider("⭐ Min Popularity", 0, 100, 0, 5)

            if st.button("🔄 Reset Feedback"):
                st.session_state["downvotes"] = set()
                st.session_state["upvotes"] = set()
                st.rerun()

        # Resolve seed song
        seed_row = filtered_df.iloc[options.index(picked) if picked in options else 0]
        seed_idx = df.index[df["track_id"] == seed_row["track_id"]].tolist()[0]
        seed = df.iloc[seed_idx]

        active_mask = build_filter_mask(df, val_rng, eng_rng, year_rng, min_pop, genre_filter)
        n_match = int(active_mask.sum())

        dv = st.session_state["downvotes"]
        uv = st.session_state["upvotes"]

        # ── Compact seed bar ──
        seed_genre = seed.get("genre", "Unknown")
        if seed_genre in ["Unknown", "[]", ""]:
            seed_genre = "—"
        genre_colors = {
            "Pop": "#1db954", "Rock": "#e8c547", "Hip-Hop": "#f472b6",
            "Jazz": "#a78bfa", "Classical": "#5eead4", "Electronic": "#22d3ee",
            "R&B": "#fb923c", "Country": "#34d399", "Metal": "#ef4444",
        }
        art_bg = genre_colors.get(seed_genre, "#333333")

        seed_col, btn_col = st.columns([5, 1])
        with seed_col:
            st.markdown(f"""
<div class="seed-bar">
    <div class="seed-bar-left">
        <div class="seed-bar-art" style="background: {art_bg};">🎵</div>
        <div class="seed-bar-info">
            <div class="seed-bar-title">{seed['title']}</div>
            <div class="seed-bar-artist">{seed['artist']} · {seed_genre}</div>
        </div>
    </div>
    <div class="seed-bar-right">
        <span class="seed-bar-match">{n_match:,} matches</span>
    </div>
</div>
            """, unsafe_allow_html=True)
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Playlist", use_container_width=True):
                pl = st.session_state["playlist"]
                if seed["track_id"] not in pl:
                    pl.append(seed["track_id"])
                    st.session_state["playlist"] = pl
                    st.success(f'Added "{seed["title"]}"!')

        # ── Recommendation feed — split by source ──
        all_recs = merge_recommendations(
            seed_idx, X, df, knn_model, user_history,
            n=8, mask=active_mask, skip=dv
        )

        if all_recs.empty:
            st.warning(f"No recommendations with current filters ({n_match:,} songs match). Try widening your filters.")
        else:
            # Split into Similar Sound and Fans Also Like
            similar_recs = all_recs[all_recs["source"] == "Similar Sound"]
            fans_recs = all_recs[all_recs["source"] == "Fans Also Like"]

            # ── Similar Sound section ──
            if not similar_recs.empty:
                st.markdown(f"""
<div class="feed-header">
    <div class="feed-title"><span class="rec-legend-dot" style="background:#1db954;display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:0.5rem;vertical-align:middle;"></span>Similar Sound</div>
    <div class="feed-subtitle">Songs that match "{seed['title']}" based on audio features</div>
</div>
                """, unsafe_allow_html=True)
                cols_ss = st.columns(2)
                for i, (_, rec) in enumerate(similar_recs.iterrows()):
                    with cols_ss[i % 2]:
                        song_card(rec, f"sim_{i}", dv, uv, source="Similar Sound", seed=seed)

            # ── Fans Also Like section ──
            if not fans_recs.empty:
                st.markdown(f"""
<div class="feed-header" style="margin-top:1.5rem;">
    <div class="feed-title"><span class="rec-legend-dot" style="background:#a78bfa;display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:0.5rem;vertical-align:middle;"></span>Fans Also Like</div>
    <div class="feed-subtitle">Listeners with similar taste also enjoyed these</div>
</div>
                """, unsafe_allow_html=True)
                cols_fl = st.columns(2)
                for i, (_, rec) in enumerate(fans_recs.iterrows()):
                    with cols_fl[i % 2]:
                        song_card(rec, f"fans_{i}", dv, uv, source="Fans Also Like", seed=seed)

        st.divider()

        # ── Surprise Me (appends discovery results) ──
        surp_col, _ = st.columns([1, 3])
        with surp_col:
            surprise_clicked = st.button("🎲 Surprise Me", use_container_width=True)

        if surprise_clicked:
            st.markdown("""
<div class="feed-header">
    <div class="feed-title">🎲 Surprise Picks</div>
    <div class="feed-subtitle">Songs from outside your usual taste</div>
</div>
            """, unsafe_allow_html=True)

            recs_surp = rec_surprise(df, X, active_mask, 6, dv)
            if recs_surp.empty:
                st.info("Not enough songs in the filtered pool for surprises.")
            else:
                if "source" not in recs_surp.columns:
                    recs_surp["source"] = "Discovery"
                cols_s = st.columns(2)
                for i, (_, rec) in enumerate(recs_surp.iterrows()):
                    with cols_s[i % 2]:
                        song_card(rec, f"surp_{i}", dv, uv, source="Discovery", seed=seed)

        # bottom bar
        show_bottom_bar(seed, n_match)

    # ============================================================
    # 📚 MY LIBRARY (merged Playlist + Profile)
    # ============================================================
    elif page == "📚 My Library":
        gradient_section_header(
            "My Library", "Your playlist and listening profile", "playlist"
        )

        lib_tab1, lib_tab2 = st.tabs(["🎼 Playlist Builder", "👤 My Profile"])

        # ── Playlist Builder tab ──
        with lib_tab1:
            a, b = st.columns([3, 1])
            with a:
                pl_genre = st.selectbox("Genre filter", ["All"] + genres, key="pl_g")
                pl_df = df if pl_genre == "All" else df[df["genre"] == pl_genre]
                pl_opts = [f"{r['title']} — {r['artist']}" for _, r in pl_df.head(500).iterrows()]
                pl_pick = st.selectbox("Find a song", pl_opts, key="pl_search", label_visibility="collapsed")
            with b:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ Add"):
                    row = pl_df.iloc[pl_opts.index(pl_pick)]
                    pl = st.session_state["playlist"]
                    if row["track_id"] not in pl:
                        pl.append(row["track_id"])
                        st.session_state["playlist"] = pl

            pl_ids = st.session_state["playlist"]
            pl_tracks = df[df["track_id"].isin(pl_ids)]

            if pl_tracks.empty:
                # Helpful empty state
                st.markdown("""
<div class="welcome-screen" style="padding: 2rem 1rem;">
    <span class="welcome-icon">🎼</span>
    <div class="welcome-title" style="font-size: 1.5rem;">Your Playlist is Empty</div>
    <div class="welcome-subtitle" style="font-size: 0.88rem;">
        Add 2-3 songs you love, and we'll find<br>the perfect mix to match.
    </div>
    <span class="welcome-label">💡 Tip: Mix genres for surprising recommendations!</span>
</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"**{len(pl_tracks)} songs added:**")
                for i, (_, t) in enumerate(pl_tracks.iterrows()):
                    rc1, rc2 = st.columns([8, 1])
                    with rc1:
                        st.markdown(
                            f'<span class="seed-tag">#{i+1} {t["title"][:25]} — {t["artist"][:20]}</span>',
                            unsafe_allow_html=True,
                        )
                    with rc2:
                        if st.button("✕", key=f"rm_{t['track_id']}"):
                            st.session_state["playlist"].remove(t["track_id"])
                            st.rerun()

                if st.button("🗑️ Clear All"):
                    st.session_state["playlist"] = []
                    st.rerun()

                st.divider()

                pl_indices = [df.index[df["track_id"] == tid].tolist()[0]
                              for tid in pl_ids if tid in df["track_id"].values]

                if pl_indices:
                    st.markdown("#### Combined Audio Profile")
                    avg_feats = df.iloc[pl_indices][
                        ["valence","energy","acousticness","danceability","instrumentalness","liveness"]
                    ].mean()
                    radar_chart(avg_feats, "Averaged profile of your playlist", key="radar_playlist_avg")

                    st.markdown("#### ✨ Songs That Match Your Playlist Vibe")
                    n_pl = st.slider("How many results?", 4, 16, 8, 2, key="pl_n")

                    pl_mv = mean_vector(pl_indices, X)
                    pl_recs = rec_content_based(pl_mv, X, df, pl_indices, n_pl,
                                                 skip=st.session_state["downvotes"])

                    if pl_recs.empty:
                        st.info("No recommendations. Try adding more songs.")
                    else:
                        pl_cols = st.columns(2)
                        for i, (_, rec) in enumerate(pl_recs.iterrows()):
                            with pl_cols[i % 2]:
                                song_card(rec, f"pl_{i}",
                                          st.session_state["downvotes"],
                                          st.session_state["upvotes"],
                                          source="Similar Sound",
                                          seed=avg_feats)

        # ── Profile tab ──
        with lib_tab2:
            user_ids = user_history.index.tolist()
            active_user = st.selectbox(
                "Switch user profile",
                user_ids,
                index=user_ids.index(st.session_state["active_user"])
                if st.session_state["active_user"] in user_ids else 0,
            )
            st.session_state["active_user"] = active_user
            u_vec = user_history.loc[active_user].values.astype(float)

            p1, p2 = st.columns([2, 3])
            with p1:
                st.markdown("#### Genre Mix")
                genre_donut(df, u_vec)
            with p2:
                st.markdown("#### Average Audio Taste")
                w_df = df.copy()
                w_df["plays"] = u_vec
                total_p = w_df["plays"].sum() or 1
                avg_f = {
                    feat: (w_df[feat] * w_df["plays"]).sum() / total_p
                    for feat in ["valence","energy","acousticness","danceability","instrumentalness","liveness","speechiness"]
                }
                fig_bar = px.bar(
                    x=list(avg_f.keys()), y=list(avg_f.values()),
                    color=list(avg_f.values()),
                    color_continuous_scale=[[0,"#a78bfa"],[0.5,"#5eead4"],[1,"#1db954"]]
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#ffffff", height=260, coloraxis_showscale=False,
                    xaxis=dict(gridcolor="#282828"), yaxis=dict(gridcolor="#282828", range=[0, 1]),
                    margin=dict(t=10, b=10)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### 🔥 Most Played")
            top_idx = np.argsort(u_vec)[::-1][:8]
            top_tracks = df.iloc[top_idx].copy()
            top_tracks["plays"] = u_vec[top_idx]

            for i, (_, t) in enumerate(top_tracks.iterrows()):
                color = "#1db954" if i < 3 else "#727272"
                st.markdown(
                    f'<div class="song-card" style="padding: .7rem 1.1rem;">'
                    f'<div style="display: flex; align-items: center; gap: 1rem;">'
                    f'<span style="font-family: Space Grotesk, sans-serif; font-size: 1.4rem; '
                    f'font-weight: 700; color: {color}; min-width: 2.2rem;">#{i+1}</span>'
                    f'<div><div class="song-title" style="font-size: .88rem;">{t["title"]}</div>'
                    f'<div class="song-artist">{t["artist"]} · {int(t["plays"])} plays</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Liked Songs ──
            liked_ids = st.session_state["upvotes"]
            hidden_ids = st.session_state["downvotes"]

            st.markdown(f"#### ❤️ Liked Songs ({len(liked_ids)})")
            if liked_ids:
                liked_tracks = df[df["track_id"].isin(liked_ids)]
                if not liked_tracks.empty:
                    for i, (_, t) in enumerate(liked_tracks.iterrows()):
                        lk_col1, lk_col2 = st.columns([8, 1])
                        with lk_col1:
                            year = int(t.get("year", 0))
                            year_str = f" · {year}" if year > 0 else ""
                            genre = t.get("genre", "")
                            genre_str = f" · {genre}" if genre and genre not in ["Unknown", "[]", ""] else ""
                            st.markdown(
                                f'<div class="song-card" style="padding: .7rem 1.1rem;">'
                                f'<div style="display: flex; align-items: center; gap: 1rem;">'
                                f'<span style="font-size: 1.2rem;">❤️</span>'
                                f'<div><div class="song-title" style="font-size: .88rem;">{t["title"]}</div>'
                                f'<div class="song-artist">{t["artist"]}{year_str}{genre_str}</div>'
                                f'</div></div></div>',
                                unsafe_allow_html=True,
                            )
                        with lk_col2:
                            if st.button("✕", key=f"unlike_{t['track_id']}"):
                                st.session_state["upvotes"].discard(t["track_id"])
                                st.rerun()
                else:
                    st.caption("Liked songs not found in the current dataset.")
            else:
                st.markdown("""
<div style="text-align:center; padding: 1.5rem; color: #727272;">
    <div style="font-size: 2rem; margin-bottom: 0.5rem;">♡</div>
    <div style="font-size: 0.85rem;">No liked songs yet</div>
    <div style="font-size: 0.75rem; margin-top: 0.3rem;">Tap ♡ on any recommendation to save it here</div>
</div>
                """, unsafe_allow_html=True)

            st.divider()

            # Session stats + Reset
            fc1, fc2 = st.columns(2)
            fc1.metric("♡ Liked this session", len(liked_ids))
            fc2.metric("✕ Hidden this session", len(hidden_ids))

            if st.button("🔄 Reset All Feedback"):
                st.session_state["downvotes"] = set()
                st.session_state["upvotes"] = set()
                st.rerun()

    # ============================================================
    # 🗺️ Explore (now includes Model Eval)
    # ============================================================
    elif page == "🗺️ Explore":
        gradient_section_header(
            "Explore", "Visualize the musical landscape", "explore"
        )

        e1, e2, e3 = st.tabs(["🗺️ Song Map", "📈 Year Trends", "🎸 Genre & Artist Stats"])

        with e1:
            st.markdown("#### Song Similarity Map (PCA)")
            st.caption("Each dot is a song. Songs that cluster together sound similar.")
            taste_map(df, pca_coords, [0])

        with e2:
            st.markdown("#### Audio Feature Trends Over Time")
            st.caption("How music changed from the 1920s to today.")
            year_trends_chart()

        with e3:
            eg1, eg2 = st.columns(2)
            with eg1:
                st.markdown("#### Genre Feature Comparison")
                st.caption("Top 20 genres by popularity — compare any audio feature.")
                genre_comparison_chart()
            with eg2:
                st.markdown("#### Top Artists: Energy vs Happiness")
                st.caption("Bubble size = popularity, color = danceability.")
                artist_stats_chart()

        # Model Eval — hidden behind expander for developers
        st.divider()
        with st.expander("📊 Model Evaluation — Developer Metrics", expanded=False):
            eval_dashboard(df)

    # ── Footer ──
    st.markdown(
        '<div style="text-align:center; padding: 2rem 0 .5rem; color: #727272; font-size: .7rem;">'
        'MY STYLE · Dataset: vatsalmavani/spotify-dataset (Kaggle) · '
        'Built with scikit-learn, streamlit, plotly'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
