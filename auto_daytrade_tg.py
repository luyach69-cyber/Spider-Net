import os
import csv
import time
import requests
import schedule
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dtime, timedelta
from FinMind.data import DataLoader

# =====================================================================
# 1. Telegram 推播設定 (自動讀取環境變數或 GitHub Secrets)
# =====================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def send_telegram(msg):
    """發送 Telegram 推播通知至手機"""
    print(f"\n[Telegram 推播] >>>\n{msg}\n")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("未設定 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，略過雲端發送。")
        return
    url = f"https://api.line.me/..." if False else f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg
    }
    try:
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code != 200:
            print(f"Telegram 推播失敗，代碼: {resp.status_code}, 內容: {resp.text}")
    except Exception as e:
        print(f"Telegram 連線異常: {e}")

# =====================================================================
# 2. 監控股票池與族群對應設定
# =====================================================================
SECTOR_MAP = {
    '光通訊': ['3081.TWO', '4979.TW', '3450.TW', '2455.TW'],
    '散熱':   ['3017.TW', '3324.TWO', '3653.TW'],
    '伺服器': ['2382.TW', '2317.TW', '6669.TW', '3231.TW'],
    '半導體': ['2330.TW', '2454.TW', '3443.TW', '3661.TW']
}

ALL_SYMBOLS = [sym for sublist in SECTOR_MAP.values() for sym in sublist]

# =====================================================================
# 3. CSV 交易日誌模組
# =====================================================================
class TradeLogger:
    def __init__(self, filename="daytrade_log.csv"):
        self.filename = filename
        if not os.path.exists(self.filename):
            headers = ["日期", "代號", "進場時間", "進場參考價", "出場時間", "出場價", "盤中最高價", "毛損益(%)", "淨損益(%)", "狀態與原因"]
            with open(self.filename, mode='w', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(headers)

    def log(self, symbol, entry_t, entry_p, exit_t, exit_p, peak_p, reason):
        cost_pct = 0.35 # 當沖手續費與證交稅預估
        gross = ((exit_p - entry_p) / entry_p) * 100
        net = gross - cost_pct
        row = [
            entry_t.strftime("%Y-%m-%d"), symbol, entry_t.strftime("%H:%M:%S"),
            round(entry_p, 2), exit_t.strftime("%H:%M:%S") if exit_t else "",
            round(exit_p, 2) if exit_p else "", round(peak_p, 2),
            round(gross, 2), round(net, 2), reason
        ]
        with open(self.filename, mode='a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(row)

# =====================================================================
# 4. 台股跳檔價值 (Tick Size) 運算
# =====================================================================
def get_tw_tick_size(price):
    if price < 10: return 0.01
    elif price < 50: return 0.05
    elif price < 100: return 0.10
    elif price < 500: return 0.50
    elif price < 1000: return 1.00
    else: return 5.00

def check_tick_advantage(price):
    tick = get_tw_tick_size(price)
    tick_pct = (tick / price) * 100
    is_favorable = tick_pct >= 0.25
    is_sweet = (100 <= price <= 105) or (500 <= price <= 520)
    return is_favorable, tick_pct, tick, is_sweet

# =====================================================================
# 5. 每日盤後選股模組 (GitHub Actions 執行目標)
# =====================================================================
def fetch_chips(stock_id):
    api = DataLoader()
    start_d = (datetime.now() - timedelta(days=12)).strftime('%Y-%m-%d')
    try:
        df = api.taiwan_stock_institutional_investors(stock_id=stock_id, start_date=start_d)
        if df.empty: return None
        df['net'] = df['buy'] - df['sell']
        return df.pivot_table(index='date', columns='name', values='net', aggfunc='sum').fillna(0)
    except Exception:
        return None

def run_daily_selection():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動台股當沖選股流程...")
    candidates = []
    
    for sym in ALL_SYMBOLS:
        sid = sym.replace('.TW', '').replace('.TWO', '')
        pivot = fetch_chips(sid)
        if pivot is None or len(pivot) < 3:
            continue
        
        recent = pivot.tail(3)
        f_col = [c for c in recent.columns if '外資' in c or 'Foreign' in c]
        t_col = [c for c in recent.columns if '投信' in c or 'Investment_Trust' in c]
        f_streak = (recent[f_col[0]] > 0).all() if f_col else False
        t_streak = (recent[t_col[0]] > 0).all() if t_col else False

        if not (f_streak or t_streak):
            continue

        try:
            df = yf.Ticker(sym).history(period="3mo")
            if len(df) < 40: continue
            
            df['MA20'] = df['Close'].rolling(20).mean()
            df['Slope20'] = df['MA20'].diff()
            
            tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            atr_pct = (atr14 / df['Close'].iloc[-1]) * 100

            latest = df.iloc[-1]
            vol_lots = latest['Volume'] / 1000

            if (latest['Close'] > latest['MA20']) and (latest['Slope20'] > 0) and (vol_lots >= 2000) and (atr_pct >= 3.0):
                score = (40 if (f_streak and t_streak) else 25) + int(atr_pct * 5)
                candidates.append({
                    'symbol': sym,
                    'name': sid,
                    'close': round(latest['Close'], 2),
                    'volume': int(vol_lots),
                    'atr_pct': round(atr_pct, 2),
                    'score': score,
                    'chip_status': "外資+投信雙連買" if (f_streak and t_streak) else ("投信連買" if t_streak else "外資連買")
                })
        except Exception:
            continue

    if not candidates:
        send_telegram("📊 【Spidernet盤後當沖選股】今日無符合高標準（多頭+法人連買+高波幅）標的。")
        return []

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top5 = candidates[:5]

    msg = f"📊 【Spidernet明日當沖精選 Top 5 標的】\n日期: {datetime.now().strftime('%Y-%m-%d')}\n"
    msg += "━━━━━━━━━━━━━━━━━━━\n"
    for idx, c in enumerate(top5, 1):
        target_p = round(c['close'] * 1.005, 1)
        sl_p = round(target_p * 0.985, 1)
        tp_p = round(target_p * 1.030, 1)
        msg += (f"{idx}. {c['name']} (昨收: {c['close']})\n"
                f"   • 籌碼: {c['chip_status']} | 振幅: {c['atr_pct']}%\n"
                f"   • 突破觀察價: {target_p}\n"
                f"   • 預計停損: {sl_p} | 預計停利: {tp_p}\n\n")
    msg += "⚠️ 叮嚀：09:15前不開倉；請配合盤中訊號手動掛單！"
    send_telegram(msg)
    return [c['symbol'] for c in top5]

if __name__ == "__main__":
    run_daily_selection()
