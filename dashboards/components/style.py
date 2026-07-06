from __future__ import annotations

import streamlit as st


def apply_global_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        h1 {
            font-size: 2.2rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.03em;
        }

        h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        .small-muted {
            color: #6b7280;
            font-size: 0.92rem;
        }

        .section-card {
            padding: 1.1rem 1.25rem;
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 18px;
            background: rgba(250, 250, 250, 0.75);
            margin-bottom: 1rem;
        }

        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(49, 51, 63, 0.14);
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.95));
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }

        .metric-label {
            font-size: 0.82rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 750;
            color: #111827;
            line-height: 1.1;
        }

        .metric-help {
            font-size: 0.78rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.12);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )