import pandas as pd

# copied from https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
def add_track_features(tracks, fps=59.94, snap_frame=10):
    """
    Add column features helpful for syncing with video data.
    """
    tracks = tracks.copy()
    tracks["game_play"] = tracks["gameKey"].astype("str") + "_" + tracks["playID"].astype("str").str.zfill(6)
    tracks["time"] = pd.to_datetime(tracks["time"])
    snap_dict = tracks.query('event == "ball_snap"').groupby("game_play")["time"].first().to_dict()
    tracks["snap"] = tracks["game_play"].map(snap_dict)
    tracks["isSnap"] = tracks["snap"] == tracks["time"]
    tracks["team"] = tracks["player"].str[0].replace("H", "Home").replace("V", "Away")
    tracks["snap_offset"] = (tracks["time"] - tracks["snap"]).astype("timedelta64[ms]") / 1_000
    # Estimated video frame
    tracks["est_frame"] = ((tracks["snap_offset"] * fps) + snap_frame).round().astype("int")
    return tracks
