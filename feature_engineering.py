import numpy as np
import pandas as pd


def _ensure_column(df, name, default=np.nan):
    if name not in df.columns:
        df[name] = default


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def build_current_features(history_rows, schema):
    if not history_rows:
        raise ValueError("No historical rows supplied.")

    df = pd.DataFrame(history_rows).copy()

    required_raw = [
        "meetingDate",
        "performanceRating",
        "preRaceAdjustedRating",
        "preRaceMasterRating",
        "timefigure",
        "prizeMoneyWon",
        "positionOfficial",
        "numberOfRunners",
        "Stakes_Win",
        "Group_Win",
        "Group1_Win",
        "horseAge",
        "draw",
        "ispDecimal",
        "betfairWinSP",
        "betfairPlaceSP",
        "ipMin",
        "distance",
        "raceClass",
        "prizeFund",
        "prizeFundWinner",
        "officialRating",
        "weightKg",
        "hotRace",
        "xWR",
        "xWM",
        "Percent_RB2",
        "leaderSectional",
        "winnerSectional",
        "distanceSectional",
        "finishingTime",
        "distanceBeaten",
    ]

    for col in required_raw:
        _ensure_column(df, col)

    df["meetingDate"] = pd.to_datetime(
        df["meetingDate"],
        errors="coerce",
        utc=True,
        dayfirst=True,
        format="mixed",
    ).dt.tz_convert(None)

    df = df[df["meetingDate"].notna()].copy()
    if df.empty:
        raise ValueError("No valid meetingDate values were found.")

    df = df.sort_values("meetingDate").reset_index(drop=True)

    numeric_cols = [
        "Stakes_Win", "Group_Win", "Group1_Win",
        "positionOfficial", "distance", "numberOfRunners",
        "prizeFund", "prizeFundWinner", "raceClass",
        "finishingTime", "leaderSectional", "winnerSectional",
        "distanceSectional", "hotRace", "horseAge", "draw",
        "ispDecimal", "betfairWinSP", "betfairPlaceSP",
        "ipMin", "performanceRating", "preRaceAdjustedRating",
        "preRaceMasterRating", "prizeMoneyWon", "timefigure",
        "distanceBeaten", "xWR", "xWM", "Percent_RB2",
        "officialRating", "weightKg",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _num(df[col])

    df["tf_clean"] = _num(df["performanceRating"])
    df.loc[(df["tf_clean"] <= 0) | (df["tf_clean"] >= 200), "tf_clean"] = np.nan

    df["pre_adj_clean"] = _num(df["preRaceAdjustedRating"])
    df.loc[(df["pre_adj_clean"] <= 0) | (df["pre_adj_clean"] >= 250), "pre_adj_clean"] = np.nan

    df["pre_master_clean"] = _num(df["preRaceMasterRating"])
    df.loc[(df["pre_master_clean"] <= 0) | (df["pre_master_clean"] >= 250), "pre_master_clean"] = np.nan

    df["tfig_clean"] = _num(df["timefigure"])
    df.loc[df["tfig_clean"] == 0, "tfig_clean"] = np.nan
    df.loc[(df["tfig_clean"] < -100) | (df["tfig_clean"] > 250), "tfig_clean"] = np.nan

    df["prize_clean"] = _num(df["prizeMoneyWon"]).fillna(0)
    df.loc[df["prize_clean"] < 0, "prize_clean"] = 0

    df["win_flag"] = (_num(df["positionOfficial"]) == 1).astype(int)
    df["place_flag"] = _num(df["positionOfficial"]).between(1, 3).astype(int)

    df["stakes_win_flag"] = (_num(df["Stakes_Win"]).fillna(0) > 0).astype(int)
    df["group_win_flag"] = (_num(df["Group_Win"]).fillna(0) > 0).astype(int)
    df["group1_win_flag"] = (_num(df["Group1_Win"]).fillna(0) > 0).astype(int)

    pos = _num(df["positionOfficial"])
    runners = _num(df["numberOfRunners"])

    df["finish_score"] = np.where(
        pos.notna() & runners.notna() & (runners > 1) & (pos > 0),
        np.clip((1 - (pos - 1) / (runners - 1)) * 100, 0, 100),
        np.nan,
    )

    isp = _num(df["ispDecimal"])
    bsp = _num(df["betfairWinSP"])
    ipmin = _num(df["ipMin"])

    df["market_drift_pct"] = np.where(
        isp.notna() & bsp.notna() & (isp > 0) & (bsp > 0),
        ((bsp - isp) / isp) * 100,
        np.nan,
    )

    df["bsp_implied_win_pct"] = np.where(
        bsp.notna() & (bsp > 1),
        100 / bsp,
        np.nan,
    )

    df["inrun_compression_pct"] = np.where(
        bsp.notna() & ipmin.notna() & (bsp > 0) & (ipmin > 0),
        np.clip(((bsp - ipmin) / bsp) * 100, -999, 100),
        np.nan,
    )

    df["runs_so_far"] = np.arange(1, len(df) + 1)
    df["wins_so_far"] = df["win_flag"].cumsum()
    df["places_so_far"] = df["place_flag"].cumsum()
    df["stakes_wins_so_far"] = df["stakes_win_flag"].cumsum()
    df["group_wins_so_far"] = df["group_win_flag"].cumsum()
    df["group1_wins_so_far"] = df["group1_win_flag"].cumsum()
    df["earnings_so_far"] = df["prize_clean"].cumsum()

    df["win_rate_so_far"] = df["wins_so_far"] / df["runs_so_far"]
    df["place_rate_so_far"] = df["places_so_far"] / df["runs_so_far"]

    df["current_tf"] = df["tf_clean"].ffill()
    df["career_high_tf"] = df["tf_clean"].cummax()

    for window in [3, 5, 10]:
        df[f"tf_avg_{window}"] = df["tf_clean"].rolling(window, min_periods=1).mean()
        df[f"tf_std_{window}"] = df["tf_clean"].rolling(window, min_periods=2).std()
        df[f"finish_avg_{window}"] = df["finish_score"].rolling(window, min_periods=1).mean()

    df["tf_prev3_avg"] = df["tf_clean"].shift(3).rolling(3, min_periods=1).mean()
    df["tf_momentum_3v3"] = df["tf_avg_3"] - df["tf_prev3_avg"]

    df["current_tfig"] = df["tfig_clean"].ffill()
    df["career_high_tfig"] = df["tfig_clean"].cummax()
    df["tfig_avg_3"] = df["tfig_clean"].rolling(3, min_periods=1).mean()
    df["tfig_avg_5"] = df["tfig_clean"].rolling(5, min_periods=1).mean()

    df["bsp_avg_3"] = bsp.rolling(3, min_periods=1).mean()
    df["market_drift_avg_3"] = df["market_drift_pct"].rolling(3, min_periods=1).mean()

    df["finish_score_current"] = df["finish_score"]

    df["previous_run_date"] = df["meetingDate"].shift(1)
    df["days_since_prev_run"] = (df["meetingDate"] - df["previous_run_date"]).dt.days

    age = _num(df["horseAge"]).replace(0, np.nan)
    df["tf_per_age"] = df["current_tf"] / age

    draw = _num(df["draw"])
    num_runners = _num(df["numberOfRunners"])
    df["relative_draw"] = np.where(
        draw.notna() & num_runners.notna() & (num_runners > 0),
        draw / num_runners,
        np.nan,
    )

    df["race_year"] = df["meetingDate"].dt.year
    df["race_month"] = df["meetingDate"].dt.month

    features = schema["features"]
    categorical_features = schema.get("categorical_features", [])

    latest = df.iloc[[-1]].copy()

    for col in features:
        if col not in latest.columns:
            latest[col] = np.nan

    X = latest[features].copy()

    for col in categorical_features:
        if col not in X.columns:
            X[col] = "__MISSING__"
        X[col] = X[col].fillna("__MISSING__").astype(str)

    current_tf = latest["current_tf"].iloc[0]

    meta = {
        "currentTF": None if pd.isna(current_tf) else float(current_tf),
        "runsSoFar": int(latest["runs_so_far"].iloc[0]),
        "stakesWinsSoFar": int(latest["stakes_wins_so_far"].iloc[0]),
        "lastRunDate": latest["meetingDate"].iloc[0].date().isoformat(),
    }

    return X, meta
