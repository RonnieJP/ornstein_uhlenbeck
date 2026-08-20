def apply_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111A22",
        plot_bgcolor="#111A22",
        font=dict(size=16),
        width=799,
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    return fig
