#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Лёгкий EDA без аргументов — экономия памяти.
- Берём только нужные колонки + семплируем (ROW_LIMIT/SAMPLE_FRAC).
- Даункаст типов (category/float32/int32).
- Строим базовые графики и гипотезы.
"""

import os, textwrap
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== Настройки памяти ====
ROW_LIMIT    = 300_000       # максимум строк (0 = не ограничивать)
SAMPLE_FRAC  = 0.10          # или доля (используется, если ROW_LIMIT = 0)
RANDOM_STATE = 42

# колонки, которые реально используем
TX_COLS = [
    "timestamp","amount","currency","channel","card_type","country",
    "is_fraud","is_card_present","is_outside_home_country"
]
FX_COLS = None  # все (их мало)

# ==== Пути ====
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TX_PATH    = os.path.join(BASE_DIR, "transaction_fraud_data.parquet")
FX_PATH    = os.path.join(BASE_DIR, "historical_currency_exchange.parquet")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR    = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close()

def write_report(text, append=False):
    path = os.path.join(REPORT_DIR, "EDA_REPORT.md")
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        f.write(text)

def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_csv(index=False) + "\n```"

def downcast(df: pd.DataFrame) -> pd.DataFrame:
    # строковые → category
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")
    # bool → uint8
    for c in df.select_dtypes("bool").columns:
        df[c] = df[c].astype("uint8")
    # float → float32
    for c in df.select_dtypes(include=["float64","float32"]).columns:
        df[c] = df[c].astype("float32")
    # int → int32
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        df[c] = df[c].astype("int32")
    return df

def load_data_lite():
    if not os.path.exists(TX_PATH): raise FileNotFoundError(TX_PATH)
    if not os.path.exists(FX_PATH): raise FileNotFoundError(FX_PATH)

    tx = pd.read_parquet(TX_PATH, columns=TX_COLS)
    fx = pd.read_parquet(FX_PATH) if FX_COLS is None else pd.read_parquet(FX_PATH, columns=FX_COLS)

    # семпл до даункаста (так быстрее)
    if ROW_LIMIT > 0 and len(tx) > ROW_LIMIT:
        tx = tx.sample(n=ROW_LIMIT, random_state=RANDOM_STATE)
    elif SAMPLE_FRAC and 0 < SAMPLE_FRAC < 1.0:
        tx = tx.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

    # типы
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True, errors="coerce")
    tx["date"]      = tx["timestamp"].dt.date
    fx["date"]      = pd.to_datetime(fx["date"]).dt.date
    tx["currency"]  = tx["currency"].astype("string").str.upper()

    # курсы → long
    fx_long = fx.melt(id_vars=["date"], var_name="currency", value_name="rate_to_usd")
    fx_long.loc[fx_long["currency"]=="USD", "rate_to_usd"] = 1.0
    fx_long["rate_to_usd"] = fx_long["rate_to_usd"].astype("float32")

    # merge
    tx = tx.merge(fx_long, on=["date","currency"], how="left")

    # amount_usd
    tx["amount"] = tx["amount"].astype("float32")
    tx["amount_usd"] = np.where(
        tx["currency"]=="USD",
        tx["amount"],
        np.where(tx["rate_to_usd"].notna() & (tx["rate_to_usd"]!=0), tx["amount"]/tx["rate_to_usd"], np.nan)
    ).astype("float32")

    # временные признаки
    ts = tx["timestamp"].dt.tz_convert(None) if tx["timestamp"].dt.tz is not None else tx["timestamp"]
    tx["hour"]      = ts.dt.hour.astype("int16")
    tx["weekday"]   = ts.dt.weekday.astype("int8")
    tx["is_weekend"]= tx["weekday"].isin([5,6]).astype("uint8")
    tx["is_night"]  = tx["hour"].isin([0,1,2,3,4,5]).astype("uint8")

    # bool → uint8 (если были bool)
    for col in ["is_fraud","is_card_present","is_outside_home_country"]:
        if col in tx.columns and tx[col].dtype=="bool":
            tx[col] = tx[col].astype("uint8")

    # строковые категориальные
    for col in ["channel","card_type","country","currency"]:
        if col in tx.columns:
            tx[col] = tx[col].astype("category")

    return downcast(tx)

def group_fraud_rate(df, by, min_count=50):
    if by not in df.columns: return pd.DataFrame()
    g = df.groupby(by, observed=True).agg(
        n=("is_fraud","size"),
        fraud_rate=("is_fraud", lambda x: x.astype("float32").mean()),
        amount_usd_median=("amount_usd","median"),
    ).reset_index()
    return g[g["n"]>=min_count].sort_values("fraud_rate", ascending=False)

def generate_hypotheses(df):
    hyps = []
    if "is_outside_home_country" in df.columns:
        out = df.loc[df["is_outside_home_country"]==1,"is_fraud"].mean()
        home= df.loc[df["is_outside_home_country"]==0,"is_fraud"].mean()
        if pd.notna(out) and pd.notna(home) and out>home:
            hyps.append(f"Вне домашней страны риск выше (вне={out:.3f}, дома={home:.3f}).")
    night = df.loc[df["is_night"]==1,"is_fraud"].mean()
    day   = df.loc[df["is_night"]==0,"is_fraud"].mean()
    if pd.notna(night) and pd.notna(day) and night>day:
        hyps.append(f"Ночью риск выше (ночь={night:.3f} > день={day:.3f}).")
    if "is_card_present" in df.columns:
        cnp = df.loc[df["is_card_present"]==0,"is_fraud"].mean()
        cp  = df.loc[df["is_card_present"]==1,"is_fraud"].mean()
        if pd.notna(cnp) and pd.notna(cp) and cnp>cp:
            hyps.append(f"CNP рискованнее (CNP={cnp:.3f} > CP={cp:.3f}).")
    if df["amount_usd"].notna().any():
        q90 = df["amount_usd"].quantile(0.9)
        hi = df.loc[df["amount_usd"]>=q90,"is_fraud"].mean()
        lo = df.loc[df["amount_usd"]< q90,"is_fraud"].mean()
        if pd.notna(hi) and pd.notna(lo) and hi>lo:
            hyps.append(f"Крупные суммы чаще фрод (top10%={hi:.3f} vs остальное={lo:.3f}).")
    hyps.append("Техгипотеза: CatBoost/XGBoost на признаках времени, географии, канала, "
                "`is_card_present`, `is_outside_home_country`, `amount_usd` + target encoding; AUC>0.9 на time-split.")
    return hyps

def run_eda():
    df = load_data_lite()
    N = len(df)
    fraud_share = df["is_fraud"].astype("float32").mean()
    unknown_fx  = df["amount_usd"].isna().mean()

    # графики (лёгкие)
    df["is_fraud"].astype(int).value_counts().sort_index().plot(kind="bar")
    plt.title("Class balance (is_fraud)")
    savefig("class_balance.png")

    df["amount_usd"].dropna().clip(upper=df["amount_usd"].quantile(0.99)).plot(kind="hist", bins=50)
    plt.title("Amount USD distribution")
    plt.xlabel("Amount (USD)"); plt.ylabel("Frequency")
    savefig("amount_hist.png")

    df.groupby("hour")["is_fraud"].mean().plot(marker="o")
    plt.title("Fraud rate by hour"); plt.xlabel("hour"); plt.ylabel("fraud_rate")
    savefig("fraud_rate_by_hour.png")

    ch = group_fraud_rate(df, "channel", min_count=30)
    if not ch.empty:
        plt.bar(ch["channel"].astype(str), ch["fraud_rate"])
        plt.title("Fraud rate by channel (>=30)"); plt.ylabel("fraud_rate")
        plt.xticks(rotation=20, ha="right")
        savefig("fraud_rate_by_channel.png")

    # отчёт
    head = textwrap.dedent(f"""
    # EDA отчёт (лёгкий режим)
    Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Использовано строк: {N:,}
    Доля мошенничества: {fraud_share:.2%}
    Транзакций без конвертации (нет курса): {unknown_fx:.2%}

    ## Ключевые графики
    - ![](figures/class_balance.png)
    - ![](figures/amount_hist.png)
    - ![](figures/fraud_rate_by_hour.png)
    - (если есть) ![](figures/fraud_rate_by_channel.png)
    """).strip()+"\n\n"
    write_report(head, append=False)

    if not ch.empty:
        write_report("## Fraud rate по каналам (>=30)\n\n", append=True)
        write_report(df_to_markdown(ch)+"\n\n", append=True)

    hyps = generate_hypotheses(df)
    write_report("## Гипотезы (продуктовые и технические)\n\n", append=True)
    for h in hyps:
        write_report(f"- {h}\n", append=True)

    write_report(textwrap.dedent("""
    ## Потенциальная ценность для организации
    - Снижение потерь: таргетная проверка high-risk паттернов (CNP, ночь, вне страны, крупные суммы).
    - Динамические лимиты/SCA по каналам с повышенным риском.
    - Поведенческий скоринг клиента (отклонения суммы/частоты/географии).
    """), append=True)

    print(f"[OK] EDA завершён. Отчёт: {os.path.join(REPORT_DIR,'EDA_REPORT.md')}")
    print(f"[OK] Графики: {FIG_DIR}")

if __name__ == "__main__":
    run_eda()
