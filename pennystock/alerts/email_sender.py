"""
SMTP email sender for stock alerts.

Uses Python's built-in smtplib — works with Gmail, Outlook, Yahoo,
or any SMTP server. Free, no third-party dependencies.

Setup (Gmail example):
  1. Enable 2FA on your Google account
  2. Create an App Password: myaccount.google.com > Security > App Passwords
  3. Use that 16-char password as ALERT_EMAIL_PASSWORD in config
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

from loguru import logger


def send_email(subject: str, body_html: str, body_text: str = "",
               smtp_server: str = "", smtp_port: int = 0,
               sender: str = "", password: str = "",
               recipient: str = "") -> bool:
    """Send an email via SMTP.

    Returns True on success, False on failure.
    """
    from pennystock.config import (
        ALERT_EMAIL_SMTP_SERVER, ALERT_EMAIL_SMTP_PORT,
        ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_RECIPIENT,
    )

    smtp_server = smtp_server or ALERT_EMAIL_SMTP_SERVER
    smtp_port = smtp_port or ALERT_EMAIL_SMTP_PORT
    sender = sender or ALERT_EMAIL_SENDER
    password = password or ALERT_EMAIL_PASSWORD
    recipient = recipient or ALERT_EMAIL_RECIPIENT

    if not all([smtp_server, sender, password, recipient]):
        logger.warning("Email not configured — skipping alert send")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")

    if body_text:
        msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        logger.info(f"Email sent: {subject}")
        return True
    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return False


# ── Pre-built alert templates ───────────────────────────────────

def send_buy_alert(picks: list) -> bool:
    """Send alert for new buy signals."""
    if not picks:
        return False

    rows = ""
    for i, p in enumerate(picks, 1):
        ss = p.get("sub_scores", {})
        ki = p.get("key_indicators", {})
        color = "#a6e3a1" if p["final_score"] >= 55 else "#f9e2af"
        rows += f"""
        <tr style="border-bottom: 1px solid #45475a;">
            <td style="padding:8px; color:{color}; font-weight:bold;">#{i}</td>
            <td style="padding:8px; font-weight:bold;">{p['ticker']}</td>
            <td style="padding:8px;">${p['price']:.2f}</td>
            <td style="padding:8px; color:{color};">{p['final_score']:.1f}</td>
            <td style="padding:8px;">
                S:{ss.get('setup',0):.0f} T:{ss.get('technical',0):.0f}
                PP:{ss.get('pre_pump',0):.0f} F:{ss.get('fundamental',0):.0f}
            </td>
            <td style="padding:8px;">{ki.get('pre_pump_confluence',0)}/7 {ki.get('pre_pump_confidence','N/A')}</td>
        </tr>"""

    html = f"""
    <div style="font-family: monospace; background:#1e1e2e; color:#cdd6f4; padding:20px;">
        <h2 style="color:#89b4fa;">BUY Signal — {len(picks)} Stock(s) Found</h2>
        <p style="color:#585b70;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <table style="width:100%; border-collapse:collapse; margin-top:10px;">
            <tr style="background:#313244;">
                <th style="padding:8px; text-align:left;">#</th>
                <th style="padding:8px; text-align:left;">Ticker</th>
                <th style="padding:8px; text-align:left;">Price</th>
                <th style="padding:8px; text-align:left;">Score</th>
                <th style="padding:8px; text-align:left;">Sub-Scores</th>
                <th style="padding:8px; text-align:left;">Pre-Pump</th>
            </tr>
            {rows}
        </table>
    </div>"""

    text = "BUY SIGNAL\n"
    for i, p in enumerate(picks, 1):
        text += f"  #{i} {p['ticker']} ${p['price']:.2f} Score:{p['final_score']:.1f}\n"

    return send_email(
        subject=f"BUY Alert: {', '.join(p['ticker'] for p in picks)}",
        body_html=html,
        body_text=text,
    )


def send_sell_alert(sells: list) -> bool:
    """Send alert for sell signals. sells = [(position, reason, description)]."""
    if not sells:
        return False

    rows = ""
    for pos, reason, description in sells:
        entry = pos["entry_price"]
        current = pos.get("current_price", entry)
        ret = ((current - entry) / entry) * 100 if entry > 0 else 0
        color = "#a6e3a1" if ret > 0 else "#f38ba8"
        rows += f"""
        <tr style="border-bottom: 1px solid #45475a;">
            <td style="padding:8px; font-weight:bold;">{pos['ticker']}</td>
            <td style="padding:8px;">${entry:.4f}</td>
            <td style="padding:8px;">${current:.4f}</td>
            <td style="padding:8px; color:{color}; font-weight:bold;">{ret:+.1f}%</td>
            <td style="padding:8px;">{reason}</td>
            <td style="padding:8px; color:#585b70;">{description}</td>
        </tr>"""

    html = f"""
    <div style="font-family: monospace; background:#1e1e2e; color:#cdd6f4; padding:20px;">
        <h2 style="color:#f38ba8;">SELL Signal — {len(sells)} Position(s)</h2>
        <p style="color:#585b70;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <table style="width:100%; border-collapse:collapse; margin-top:10px;">
            <tr style="background:#313244;">
                <th style="padding:8px; text-align:left;">Ticker</th>
                <th style="padding:8px; text-align:left;">Entry</th>
                <th style="padding:8px; text-align:left;">Current</th>
                <th style="padding:8px; text-align:left;">Return</th>
                <th style="padding:8px; text-align:left;">Trigger</th>
                <th style="padding:8px; text-align:left;">Detail</th>
            </tr>
            {rows}
        </table>
    </div>"""

    text = "SELL SIGNAL\n"
    for pos, reason, description in sells:
        entry = pos["entry_price"]
        current = pos.get("current_price", entry)
        ret = ((current - entry) / entry) * 100 if entry > 0 else 0
        text += f"  {pos['ticker']}: {ret:+.1f}% — {description}\n"

    tickers = [pos["ticker"] for pos, _, _ in sells]
    return send_email(
        subject=f"SELL Alert: {', '.join(tickers)}",
        body_html=html,
        body_text=text,
    )


def send_portfolio_summary(summary: dict, positions: list) -> bool:
    """Send daily portfolio summary email."""
    pos_rows = ""
    for pos in positions:
        entry = pos["entry_price"]
        current = pos.get("current_price", entry)
        ret = ((current - entry) / entry) * 100 if entry > 0 else 0
        color = "#a6e3a1" if ret > 0 else "#f38ba8"
        trail = "YES" if pos.get("trailing_stop_active") else ""
        pos_rows += f"""
        <tr style="border-bottom: 1px solid #45475a;">
            <td style="padding:6px;">{pos['ticker']}</td>
            <td style="padding:6px;">{pos['shares']}</td>
            <td style="padding:6px;">${entry:.4f}</td>
            <td style="padding:6px;">${current:.4f}</td>
            <td style="padding:6px; color:{color}; font-weight:bold;">{ret:+.1f}%</td>
            <td style="padding:6px;">{trail}</td>
        </tr>"""

    val_color = "#a6e3a1" if summary["total_return_pct"] >= 0 else "#f38ba8"

    html = f"""
    <div style="font-family: monospace; background:#1e1e2e; color:#cdd6f4; padding:20px;">
        <h2 style="color:#89b4fa;">Daily Portfolio Summary</h2>
        <p style="color:#585b70;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

        <div style="display:flex; gap:20px; margin:15px 0;">
            <div style="background:#313244; padding:12px 20px; border-radius:8px;">
                <div style="color:#585b70; font-size:11px;">PORTFOLIO VALUE</div>
                <div style="font-size:20px; color:{val_color}; font-weight:bold;">
                    ${summary['total_value']:,.2f}
                </div>
                <div style="color:{val_color};">{summary['total_return_pct']:+.1f}%</div>
            </div>
            <div style="background:#313244; padding:12px 20px; border-radius:8px;">
                <div style="color:#585b70; font-size:11px;">CASH</div>
                <div style="font-size:20px;">${summary['cash']:,.2f}</div>
            </div>
            <div style="background:#313244; padding:12px 20px; border-radius:8px;">
                <div style="color:#585b70; font-size:11px;">WIN RATE</div>
                <div style="font-size:20px;">{summary['win_rate']:.0f}%</div>
                <div style="color:#585b70;">{summary['total_trades']} trades</div>
            </div>
        </div>

        <h3 style="color:#cdd6f4; margin-top:20px;">Positions ({summary['num_positions']})</h3>
        <table style="width:100%; border-collapse:collapse;">
            <tr style="background:#313244;">
                <th style="padding:6px; text-align:left;">Ticker</th>
                <th style="padding:6px; text-align:left;">Shares</th>
                <th style="padding:6px; text-align:left;">Entry</th>
                <th style="padding:6px; text-align:left;">Current</th>
                <th style="padding:6px; text-align:left;">Return</th>
                <th style="padding:6px; text-align:left;">Trail</th>
            </tr>
            {pos_rows}
        </table>
    </div>"""

    text = (
        f"PORTFOLIO: ${summary['total_value']:,.2f} ({summary['total_return_pct']:+.1f}%)\n"
        f"Cash: ${summary['cash']:,.2f} | Positions: {summary['num_positions']}\n"
        f"Win Rate: {summary['win_rate']:.0f}% ({summary['total_trades']} trades)\n"
    )

    return send_email(
        subject=f"Portfolio: ${summary['total_value']:,.2f} ({summary['total_return_pct']:+.1f}%)",
        body_html=html,
        body_text=text,
    )


def send_test_email() -> bool:
    """Send a test email to verify configuration."""
    html = f"""
    <div style="font-family: monospace; background:#1e1e2e; color:#cdd6f4; padding:20px;">
        <h2 style="color:#a6e3a1;">Email Alerts Configured Successfully</h2>
        <p style="color:#585b70;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        <p>You will receive alerts for:</p>
        <ul>
            <li><span style="color:#89b4fa;">BUY signals</span> — when the algorithm finds picks</li>
            <li><span style="color:#f38ba8;">SELL signals</span> — stop-loss, take-profit, trailing stop, max hold</li>
            <li><span style="color:#f9e2af;">Daily summary</span> — portfolio status at market close</li>
        </ul>
    </div>"""

    return send_email(
        subject="Stock Analyzer — Email Alerts Active",
        body_html=html,
        body_text="Stock Analyzer email alerts configured successfully.",
    )
