"""Plotly 2D/3D scatter plot for embedding visualization."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_scatter_2d(
    coords: np.ndarray,
    labels: list[str],
    hover_texts: list[str],
    title: str = "Embedding Space",
    color_by: list[str] | None = None,
):
    """Render 2D scatter plot with hover text."""
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "label": labels,
        "text": [t[:200] for t in hover_texts],
        "color": color_by if color_by else labels,
    })

    fig = px.scatter(
        df, x="x", y="y",
        color="color",
        hover_data={"text": True, "x": False, "y": False, "color": False},
        title=title,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(height=550, legend_title="Strategy/Experiment")
    st.plotly_chart(fig, width="stretch")


def render_scatter_3d(
    coords: np.ndarray,
    labels: list[str],
    hover_texts: list[str],
    title: str = "Embedding Space (3D)",
    color_by: list[str] | None = None,
):
    """Render 3D scatter plot."""
    if coords.shape[1] < 3:
        st.warning("3D requires 3 components. Switch to 3D in settings.")
        return

    color_vals = color_by if color_by else labels
    unique_colors = list(dict.fromkeys(color_vals))
    palette = px.colors.qualitative.Plotly

    traces = []
    for i, cat in enumerate(unique_colors):
        mask = [c == cat for c in color_vals]
        idx = [j for j, m in enumerate(mask) if m]
        trace = go.Scatter3d(
            x=coords[idx, 0],
            y=coords[idx, 1],
            z=coords[idx, 2],
            mode="markers",
            name=cat,
            marker=dict(size=5, color=palette[i % len(palette)], opacity=0.8),
            text=[hover_texts[j][:200] for j in idx],
            hovertemplate="%{text}<extra></extra>",
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        height=600,
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3",
        ),
    )
    st.plotly_chart(fig, width="stretch")
