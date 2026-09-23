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
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
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

        # 法人未連續買超者過濾
        if not (f_streak or t_streak):
            continue

        try:
            df = yf.Ticker(sym).history(period="3mo")
            if len(df) < 40: continue
            
            # 計算 20MA 多頭結構
            df['MA20'] = df['Close'].rolling(20).mean()
            df['Slope20'] = df['MA20'].diff()
            
            # ATR (14日波幅)
            tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            atr_pct = (atr14 / df['Close'].iloc[-1]) * 100

            latest = df.iloc[-1]
            vol_lots = latest['Volume'] / 1000

            # 核心門檻: 站在20MA上且20MA翻揚、日均量>=2000張、ATR>=3%
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
        send_telegram("📊 【盤後當沖選股】今日無符合高標準（多頭+法人連買+高波幅）標的。")
        return []

    candidates.sort(key=lambda x: x['score'], reverse=True)
    top5 = candidates[:5]

    msg = f"📊 【明日當沖精選 Top 5 標的】\n日期: {datetime.now().strftime('%Y-%m-%d')}\n"
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

# =====================================================================
# 6. 盤中即時監控引擎 (本機常駐盯盤用)
# =====================================================================
class LiveIntradayEngine:
    def __init__(self, watch_symbols, logger: TradeLogger):
        self.watch_symbols = watch_symbols
        self.logger = logger
        self.morning_highs = {s: 0.0 for s in watch_symbols}
        self.positions = {}
        self.triggered_entries = set()

    def check_market_filter(self):
        try:
            twii = yf.Ticker("^TWII").history(period="1d", interval="1m")
            if twii.empty: return True
            return twii.iloc[-1]['Close'] >= twii.iloc[0]['Open']
        except Exception:
            return True

    def scan(self):
        now = datetime.now()
        cur_t = now.time()
        if cur_t < dtime(9, 0) or cur_t > dtime(13, 30):
            return

        market_ok = self.check_market_filter()
        try:
            data = yf.download(tickers=ALL_SYMBOLS, period="1d", interval="1m", progress=False, group_by='ticker')
        except Exception:
            return

        live_quotes = {}
        for sym in ALL_SYMBOLS:
            try:
                df = data[sym].dropna()
                if df.empty: continue
                c = df.iloc[-1]['Close']
                o = df.iloc[0]['Open']
                cum_vol = df['Volume'].sum()
                cum_val = (df['Close'] * df['Volume']).sum()
                vwap = (cum_val / cum_vol) if cum_vol > 0 else c
                
                vol_ratio = 1.0
                if len(df) >= 6:
                    prev_v = df['Volume'].iloc[-6:-1].mean()
                    vol_ratio = (df['Volume'].iloc[-1] / prev_v) if prev_v > 0 else 1.0
                
                pct = ((c - o) / o) * 100
                live_quotes[sym] = {'price': c, 'high': df.iloc[-1]['High'], 'low': df.iloc[-1]['Low'], 'open': o, 'vwap': vwap, 'vol_ratio': vol_ratio, 'pct': pct}
            except Exception:
                continue

        # 09:00 ~ 09:15 隔日沖沉澱期，記錄高點
        if dtime(9, 0) <= cur_t < dtime(9, 15):
            for sym in self.watch_symbols:
                q = live_quotes.get(sym)
                if q and q['high'] > self.morning_highs[sym]:
                    self.morning_highs[sym] = q['high']
            return

        # 09:15 ~ 11:15 監控進場訊號
        if dtime(9, 15) <= cur_t <= dtime(11, 15) and market_ok:
            for sym in self.watch_symbols:
                if sym in self.triggered_entries or sym in self.positions:
                    continue
                q = live_quotes.get(sym)
                if not q: continue

                p = q['price']
                vwap = q['vwap']
                vol_r = q['vol_ratio']
                m_high = self.morning_highs[sym]

                favorable_tick, tick_pct, tick_val, is_sweet = check_tick_advantage(p)
                if not favorable_tick:
                    continue

                if not (p >= m_high and p > vwap and vol_r >= 1.8):
                    continue

                # 族群同動性判定
                my_sector = next((sec for sec, syms in SECTOR_MAP.items() if sym in syms), None)
                if my_sector:
                    sec_syms = SECTOR_MAP[my_sector]
                    strong_cnt = sum(1 for s in sec_syms if (s in live_quotes and live_quotes[s]['pct'] >= 1.5 and live_quotes[s]['price'] >= live_quotes[s]['vwap']))
                    if strong_cnt < 2:
                        continue
                    sec_tag = f"【{my_sector}】族群強勢放量 ({strong_cnt}/{len(sec_syms)})"
                else:
                    sec_tag = "無族群資料"

                self.triggered_entries.add(sym)
                init_sl = round(p * 0.985, 2)
                init_tp = round(p * 1.030, 2)
                self.positions[sym] = {
                    'entry_time': now, 'entry_p': p, 'stop_p': init_sl,
                    'peak_p': p, 'locked_be': False
                }

                sweet_msg = "🔥【甜蜜檔位，1~2跳即獲利】" if is_sweet else f"跳一檔: +{round(tick_pct, 2)}%"
                send_telegram(
                    f"🎯 【高勝率當沖進場訊號】\n"
                    f"━━━━━━━━━━━━━━━━━━━\n"
                    f"• 標的: {sym.split('.')[0]}\n"
                    f"• 時間: {now.strftime('%H:%M:%S')}\n"
                    f"• 現價: {p} (Tick: {tick_val}元 | {sweet_msg})\n"
                    f"• VWAP: {round(vwap, 2)} | 量比: {round(vol_r, 2)}x\n"
                    f"• 族群狀況: {sec_tag}\n"
                    f"━━━━━━━━━━━━━━━━━━━\n"
                    f"📋 【手動下單指引】：\n"
                    f"1. 請手動於證券 App 買進\n"
                    f"2. 停損防守: {init_sl} (-1.5%)\n"
                    f"3. 預計停利: {init_tp} (+3.0%)"
                )

        # 持倉監控：保本、移動停利與收盤強制平倉
        closed_syms = []
        for sym, pos in self.positions.items():
            q = live_quotes.get(sym)
            if not q: continue
            
            p = q['price']
            if p > pos['peak_p']:
                pos['peak_p'] = p

            gain_pct = ((pos['peak_p'] - pos['entry_p']) / pos['entry_p']) * 100

            # 保本機制 (+1.5% 時啟動)
            if not pos['locked_be'] and gain_pct >= 1.5:
                be_price = round(pos['entry_p'] * 1.0035, 2)
                if be_price > pos['stop_p']:
                    pos['stop_p'] = be_price
                    pos['locked_be'] = True
                    send_telegram(f"🛡️ 【保本通知】{sym.split('.')[0]} 漲幅達 +{round(gain_pct, 2)}%！\n停損點已推至成本保本價 {be_price}，立於不敗之地！")

            # 移動停利 (+3.0% 以上啟動，自高點拉回 1% 出場)
            if gain_pct >= 3.0:
                trail_stop = round(pos['peak_p'] * 0.990, 2)
                if trail_stop > pos['stop_p']:
                    pos['stop_p'] = trail_stop

            # 出場判定
            if p <= pos['stop_p'] or p < q['vwap']:
                reason = "移動停利鎖利" if gain_pct >= 3.0 else ("觸發保本出場" if pos['locked_be'] else "跌破停損線")
                pnl = round(((p - pos['entry_p']) / pos['entry_p']) * 100, 2)
                self.logger.log(sym, pos['entry_time'], pos['entry_p'], now, p, pos['peak_p'], reason)
                send_telegram(
                    f"🔔 【平倉出場通知】{sym.split('.')[0]}！\n"
                    f"• 時間: {now.strftime('%H:%M:%S')}\n"
                    f"• 出場價: {p} (進場: {pos['entry_p']})\n"
                    f"• 盤中最高: {pos['peak_p']} | 毛損益: {pnl}%\n"
                    f"• 原因: {reason}，請手動賣出平倉！"
                )
                closed_syms.append(sym)
                continue

            # 13:00 尾盤強制平倉
            if cur_t >= dtime(13, 0):
                pnl = round(((p - pos['entry_p']) / pos['entry_p']) * 100, 2)
                self.logger.log(sym, pos['entry_time'], pos['entry_p'], now, p, pos['peak_p'], "尾盤不留倉強制平倉")
                send_telegram(f"⏰ 【尾盤強制平倉】{sym.split('.')[0]} 出場價: {p}，損益: {pnl}%。今日事今日畢！")
                closed_syms.append(sym)

        for s in closed_syms:
            del self.positions[s]

# =====================================================================
# 7. 主排程與常駐執行
# =====================================================================
def main():
    logger = TradeLogger()
    print("==================================================")
    print("🚀 台股高勝率當沖自動監控與 Telegram 推播系統已啟動...")
    print("• 每日 15:35 自動執行盤後選股與 Telegram 推播")
    print("• 開盤期間 09:00 ~ 13:30 每 15 秒執行盤中高勝率進出監控")
    print("==================================================")

    daily_watch = ['3081.TWO', '4979.TW', '3017.TW', '3324.TWO', '2382.TW']
    engine = LiveIntradayEngine(daily_watch, logger)

    def job_selection():
        nonlocal daily_watch, engine
        selected = run_daily_selection()
        if selected:
            daily_watch = selected
            engine = LiveIntradayEngine(daily_watch, logger)

    schedule.every().day.at("15:35").do(job_selection)

    while True:
        schedule.run_pending()
        cur_t = datetime.now().time()
        if dtime(9, 0) <= cur_t <= dtime(13, 30):
            engine.scan()
            time.sleep(15)
        else:
            time.sleep(30)

if __name__ == "__main__":
    main()
