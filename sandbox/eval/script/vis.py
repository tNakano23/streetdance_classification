import re
import sys
import os
import glob
import pandas as pd
import plotly.express as px
import streamlit as st
print(pwd := __file__)


# =========================================================
# 0. エントリポイント
# =========================================================

FILE_PATH = None

if len(sys.argv) >= 2:
    arg_path = sys.argv[1]

    if os.path.exists(arg_path):
        FILE_PATH = arg_path

# ----------------------------------------------------------
# file explorer
# ----------------------------------------------------------

if FILE_PATH is None:

    st.sidebar.header("Select Result File")

    candidate_files = sorted(
        glob.glob(
            "data/result_eval/**/summary_detail.txt",
            recursive=True,
        )
    )

    if len(candidate_files) == 0:
        st.error("summary_detail.txt not found")
        st.stop()

    FILE_PATH = st.sidebar.selectbox(
        "summary_detail.txt",
        candidate_files,
    )



st.title("Dance Motion Analysis")

# =========================================================
# 1. regex
# =========================================================

pattern_content = re.compile(
    r"""
    ^.*CSV:\s(?P<csv>.+)\n
    Valid\ samples\ \(Keyword\ Matched\):\s(?P<valid_samples>\d+\s/\s\d+)\n
    Metrics\ \(acc,\ f1\):\n
    \s+\[Genre\]\s+\d+\.\d+%\s+&\s+\d+\.\d+%\n
    \s+\[Move\]\s+\d+\.\d+%\s+&\s+\d+\.\d+%\n
    \s+\[GM\]\s+\d+\.\d+%\s+&\s+\d+\.\d+%\n
    \n
    \[GM\ per\ class\]\n
    (?P<gm_per_classes>.*)$
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)

pattern_pet = re.compile(
    r"""
    ^(?P<g_m>.+?)
    :
    \sacc=(?P<acc>\d+\.\d+)%
    ,
    \sf1=(?P<f1>\d+\.\d+)%
    ,
    \sn=(?P<n>\d+)
    \s\|\s
    (?P<inner>\[.*\])
    $
    """,
    re.VERBOSE,
)


# =========================================================
# 2. parser
# =========================================================

def parse_each_content(
    file_path, sep="=================================================="
):
    with open(file_path, "r") as f:
        content = f.read()
    contents = content.split(sep)
    contents = [_ for _ in contents if len(_) > 20]
    return contents


def parse_each_tag(content, pattern=pattern_content):
    match = pattern.match(content)
    result_dict = match.groupdict() if match else {}
    return result_dict


def parse_gm_per_classes(item, pattern=pattern_pet):
    result_list = []

    for line in item.splitlines():
        match = pattern.match(line.strip())

        if not match:
            continue
        result_dict = match.groupdict()
        result_dict["acc"] = float(result_dict["acc"])
        result_dict["f1"] = float(result_dict["f1"])
        result_dict["n"] = int(result_dict["n"])

        genre, move = result_dict["g_m"].rsplit("_", 1)
        result_dict["genre"] = genre
        result_dict["move"] = move

        # ---------------------------------
        # inner
        # ---------------------------------

        inner_matches = re.findall(
            r"【(.*?):(\d+)件】",
            result_dict["inner"],
        )

        inner_list = []
        for gm, n in inner_matches:
            genre_i, move_i = gm.rsplit("_", 1)

            inner_list.append(
                {
                    "g_m": gm,
                    "genre": genre_i,
                    "move": move_i,
                    "count": int(n),
                }
            )

        result_dict["inner"] = inner_list
        result_list.append(result_dict)

    return result_list


# =========================================================
# 3. flatten
# =========================================================

def flatten_result(dict_list):

    rows = []
    for item in dict_list:
        csv_name = item["csv"]

        for gm_item in item["gm_per_classes"]:
            rows.append(
                {
                    "csv": csv_name,
                    "g_m": gm_item["g_m"],
                    "genre": gm_item["genre"],
                    "move": gm_item["move"],
                    "acc": gm_item["acc"],
                    "f1": gm_item["f1"],
                    "n": gm_item["n"],
                    "inner": gm_item["inner"],
                }
            )

    return rows


# =========================================================
# 4. load
# =========================================================

contents = parse_each_content(FILE_PATH)

dict_list = []
for content in contents:
    result_dict = parse_each_tag(content)

    gm_result = parse_gm_per_classes(
        result_dict["gm_per_classes"]
    )

    result_dict["gm_per_classes"] = gm_result
    dict_list.append(result_dict)
rows = flatten_result(dict_list)
df = pd.DataFrame(rows)


# =========================================================
# 5. sidebar
# =========================================================

st.sidebar.title("Filter")

genre_filter = st.sidebar.multiselect(
    "Genre",
    sorted(df["genre"].unique()),
)

csv_filter = st.sidebar.multiselect(
    "CSV",
    sorted(df["csv"].unique()),
)

if genre_filter:
    df = df[df["genre"].isin(genre_filter)]

if csv_filter:
    df = df[df["csv"].isin(csv_filter)]


# =========================================================
# 6. overview
# =========================================================

st.header("Overview")
weighted_acc = (df["acc"] * df["n"]).sum() / df["n"].sum()
weighted_f1 = (df["f1"] * df["n"]).sum() / df["n"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Weighted ACC", f"{weighted_acc:.2f}%")
col2.metric("Weighted F1", f"{weighted_f1:.2f}%")
col3.metric("GM Classes", len(df))


# ----------------------------------------------------------
# genre accuracy
# ----------------------------------------------------------

st.header("Genre Accuracy")

genre_df = (
    df.groupby("genre")
    .apply(lambda x: (x["acc"] * x["n"]).sum() / x["n"].sum())
    .reset_index(name="weighted_acc")
)

fig_genre = px.bar(
    genre_df,
    x="genre",
    y="weighted_acc",
)

st.plotly_chart(fig_genre, width=True)

# ----------------------------------------------------------
# move accuracy
# ----------------------------------------------------------

st.header("Move Accuracy")

move_df = (
    df.groupby("move")
    .apply(lambda x: (x["acc"] * x["n"]).sum() / x["n"].sum())
    .reset_index(name="weighted_acc")
)

move_df = move_df.sort_values("weighted_acc", ascending=False)

fig_move = px.bar(
    move_df,
    x="move",
    y="weighted_acc",
)

st.plotly_chart(fig_move, width=True)

st.dataframe(move_df, width=True)


# ----------------------------------------------------------
# GM ranking
# ----------------------------------------------------------

st.header("GM Ranking")

sort_key = st.selectbox("Sort by", ["acc", "f1", "n"])

rank_df = df.sort_values(sort_key, ascending=False)

st.dataframe(
    rank_df[
        [
            "g_m",
            "genre",
            "move",
            "acc",
            "f1",
            "n",
        ]
    ],
    width=True,
)


# ----------------------------------------------------------
# GM detail
# ----------------------------------------------------------

st.header("GM Detail")

selected_gm = st.selectbox("Select GM", sorted(df["g_m"].unique()))

target = df[df["g_m"] == selected_gm].iloc[0]

st.write(target)


# =========================================================
# confusion analysis
# =========================================================

st.header("Confusion Analysis")

conf_rows = []

for _, row in df.iterrows():
    true_label = row["g_m"]

    for pred in row["inner"]:
        conf_rows.append(
            {
                "true": true_label,
                "pred": pred["g_m"],
                "true_genre": row["genre"],
                "pred_genre": pred["genre"],
                "true_move": row["g_m"],
                "pred_move": pred["g_m"],
                "count": pred["count"],
            }
        )

conf_df = pd.DataFrame(conf_rows)


# =========================================================
# top mistakes
# =========================================================

st.subheader("Top Mistakes")

mistakes = conf_df[conf_df["true"] != conf_df["pred"]]

mistakes = (
    mistakes.groupby(["true", "pred"])["count"]
    .sum()
    .reset_index()
    .sort_values("count", ascending=False)
)

st.dataframe(mistakes.head(30), width=True)


# =========================================================
# rename_csv_filter
# =========================================================
csv_filter_parallel_dict = {
    "1_choices_given.csv": "choices_given",
    "2_video2video.csv": "video-FT",
    "3_beat2beat.csv": "beat_clip",
    "4_int200ms2int200ms.csv": "interval_clip(200ms)",
    "5_int500ms2int500ms.csv": "interval_clip(500ms)",
}

csv_name = "all"

if len(csv_filter) > 0:
    csv_name = ",".join(
        csv_filter_parallel_dict[x]
        for x in csv_filter
        if x in csv_filter_parallel_dict
    )

# =========================================================
# genre heatmap
# =========================================================

st.subheader("Genre Confusion Heatmap")

genre_conf = conf_df.groupby(["true_genre", "pred_genre"])["count"].sum().reset_index()

pivot_genre = genre_conf.pivot_table(
    index="true_genre",
    columns="pred_genre",
    values="count",
    fill_value=0,
)

fig_genre_heat = px.imshow(
    pivot_genre,
    text_auto=True,
    aspect="auto",
)

title_text_g = f"genre heatmap({csv_name})"

fig_genre_heat.update_layout(
    title={
        "text": title_text_g,
        "x": 0.5,
        "xanchor": "center",
    }
)

# -------------------------------------------------
# diagonal highlight (robust version)
# -------------------------------------------------

x_labels = list(pivot_genre.columns)
y_labels = list(pivot_genre.index)

for genre in y_labels:
    # pred側に存在しない場合はskip
    if genre not in x_labels:
        continue

    x = x_labels.index(genre)
    y = y_labels.index(genre)

    fig_genre_heat.add_shape(
        type="circle",
        x0=x - 0.45,
        y0=y - 0.45,
        x1=x + 0.45,
        y1=y + 0.45,
        line=dict(width=3),
    )

st.plotly_chart(fig_genre_heat, width=True)


# =========================================================
# move confusion heatmap
# =========================================================

st.subheader("Move Confusion Heatmap")

move_conf = conf_df.groupby(["true_move", "pred_move"])["count"].sum().reset_index()

pivot_move = move_conf.pivot_table(
    index="true_move",
    columns="pred_move",
    values="count",
    fill_value=0,
)

fig_move_heat = px.imshow(pivot_move, text_auto=True, aspect="auto")

title_text_m = f"Move Confusion Heatmap({csv_name})"
fig_move_heat.update_layout(
    title={
        "text": title_text_m,
        "x": 0.5,
        "xanchor": "center",
    }
)

# -------------------------------------------------
# diagonal highlight (robust version)
# -------------------------------------------------

x_labels = list(pivot_move.columns)
y_labels = list(pivot_move.index)

for move in y_labels:
    # pred側に存在しない場合はskip
    if move not in x_labels:
        continue

    x = x_labels.index(move)
    y = y_labels.index(move)

    fig_move_heat.add_shape(
        type="circle",
        x0=x - 0.45,
        y0=y - 0.45,
        x1=x + 0.45,
        y1=y + 0.45,
        line=dict(width=3),
    )

st.plotly_chart(fig_move_heat, width=True)
