#!/usr/bin/env python3
"""Fetch daily download and engagement stats for Driftcheck and send via Resend email API."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

TARGET_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "boobalan_arjunan@composermind.com")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "Driftcheck Stats <onboarding@resend.dev>")


def fetch_stats():
    # 1. PyPI Stats
    total_with_mirrors, total_without_mirrors = 0, 0
    last_day, last_week = 0, 0
    pypi_history = []
    try:
        req = urllib.request.Request(
            "https://pypistats.org/api/packages/driftcheck/overall",
            headers={"User-Agent": "Driftcheck-Reporter"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode())
            rows = d.get("data", [])
            total_with_mirrors = sum(r["downloads"] for r in rows if r["category"] == "with_mirrors")
            total_without_mirrors = sum(r["downloads"] for r in rows if r["category"] == "without_mirrors")
            pypi_history = [r for r in rows if r["category"] == "without_mirrors"][-7:]
    except Exception as e:
        print(f"[WARN] PyPI overall stats failed: {e}")

    try:
        req = urllib.request.Request(
            "https://pypistats.org/api/packages/driftcheck/recent",
            headers={"User-Agent": "Driftcheck-Reporter"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.loads(resp.read().decode())
            last_day = d.get("data", {}).get("last_day", 0)
            last_week = d.get("data", {}).get("last_week", 0)
    except Exception as e:
        print(f"[WARN] PyPI recent stats failed: {e}")

    # 2. GitHub Stats
    gh_stars, gh_forks, gh_issues = 0, 0, 0
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/BoobalanArjunan/Driftcheck",
            headers={"User-Agent": "Driftcheck-Reporter"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            gh = json.loads(resp.read().decode())
            gh_stars = gh.get("stargazers_count", 0)
            gh_forks = gh.get("forks_count", 0)
            gh_issues = gh.get("open_issues_count", 0)
    except Exception as e:
        print(f"[WARN] GitHub stats failed: {e}")

    # 3. Hugging Face Space
    hf_likes, hf_stage = 0, "Active"
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/spaces/boobalanit/driftcheck",
            headers={"User-Agent": "Driftcheck-Reporter"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            hf = json.loads(resp.read().decode())
            hf_likes = hf.get("likes", 0)
            hf_stage = hf.get("runtime", {}).get("stage", "Sleeping")
    except Exception as e:
        print(f"[WARN] Hugging Face stats failed: {e}")

    return {
        "total_with_mirrors": total_with_mirrors,
        "total_without_mirrors": total_without_mirrors,
        "last_day": last_day,
        "last_week": last_week,
        "pypi_history": pypi_history,
        "gh_stars": gh_stars,
        "gh_forks": gh_forks,
        "gh_issues": gh_issues,
        "hf_likes": hf_likes,
        "hf_stage": hf_stage,
        "date_str": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def build_html_email(stats: dict) -> str:
    history_rows = ""
    for row in reversed(stats["pypi_history"]):
        history_rows += f"""
        <tr>
          <td style="padding: 8px 12px; border-bottom: 1px solid #27272a; color: #a1a1aa; font-family: monospace;">{row['date']}</td>
          <td style="padding: 8px 12px; border-bottom: 1px solid #27272a; text-align: right; font-weight: bold; color: #38bdf8; font-family: monospace;">{row['downloads']}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Driftcheck Daily Analytics</title>
</head>
<body style="margin: 0; padding: 24px; background-color: #09090b; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #f4f4f5;">
  <div style="max-width: 600px; margin: 0 auto; background: #18181b; border: 1px solid #27272a; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
    
    <!-- Header -->
    <div style="background: linear-gradient(135deg, #0284c7, #6366f1); padding: 24px; text-align: center;">
      <h1 style="margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.5px; color: #ffffff;">📊 Driftcheck Daily Analytics</h1>
      <p style="margin: 6px 0 0 0; font-size: 13px; color: #e0e7ff;">Automated Ecosystem Performance Report &bull; {stats['date_str']}</p>
    </div>

    <!-- Core Metrics -->
    <div style="padding: 24px;">
      <table style="width: 100%; border-collapse: separate; border-spacing: 12px 0;">
        <tr>
          <td style="background: #27272a; border-radius: 8px; padding: 16px; text-align: center; width: 33%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">Yesterday</div>
            <div style="font-size: 26px; font-weight: 800; color: #38bdf8; margin: 6px 0;">+{stats['last_day']}</div>
            <div style="font-size: 11px; color: #a1a1aa;">Direct Installs</div>
          </td>
          <td style="background: #27272a; border-radius: 8px; padding: 16px; text-align: center; width: 33%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">Last 7 Days</div>
            <div style="font-size: 26px; font-weight: 800; color: #a855f7; margin: 6px 0;">{stats['last_week']}</div>
            <div style="font-size: 11px; color: #a1a1aa;">Direct Installs</div>
          </td>
          <td style="background: #27272a; border-radius: 8px; padding: 16px; text-align: center; width: 33%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">All-Time</div>
            <div style="font-size: 26px; font-weight: 800; color: #22c55e; margin: 6px 0;">{stats['total_with_mirrors']}</div>
            <div style="font-size: 11px; color: #a1a1aa;">With Mirrors</div>
          </td>
        </tr>
      </table>

      <!-- Secondary Metrics -->
      <table style="width: 100%; border-collapse: separate; border-spacing: 12px 0; margin-top: 12px;">
        <tr>
          <td style="background: #27272a; border-radius: 8px; padding: 16px; text-align: center; width: 50%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">GitHub Repo</div>
            <div style="font-size: 22px; font-weight: 700; color: #fbbf24; margin: 6px 0;">⭐ {stats['gh_stars']} &bull; {stats['gh_forks']} 🍴</div>
            <div style="font-size: 11px; color: #a1a1aa;">Issues: {stats['gh_issues']}</div>
          </td>
          <td style="background: #27272a; border-radius: 8px; padding: 16px; text-align: center; width: 50%; border: 1px solid #3f3f46;">
            <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px;">Hugging Face Space</div>
            <div style="font-size: 22px; font-weight: 700; color: #f59e0b; margin: 6px 0;">{stats['hf_likes']} ❤️</div>
            <div style="font-size: 11px; color: #a1a1aa;">Status: {stats['hf_stage']}</div>
          </td>
        </tr>
      </table>

      <!-- PyPI History Table -->
      <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #cbd5e1; margin: 24px 0 10px 0; font-weight: 700;">Recent Daily Direct Installs (Last 7 Days)</h3>
      <table style="width: 100%; border-collapse: collapse; background: #202023; border-radius: 8px; overflow: hidden; border: 1px solid #27272a; font-size: 13px;">
        <thead>
          <tr style="background: #27272a;">
            <th style="padding: 10px 12px; text-align: left; color: #94a3b8; font-weight: 600; font-size: 11px; text-transform: uppercase;">Date</th>
            <th style="padding: 10px 12px; text-align: right; color: #94a3b8; font-weight: 600; font-size: 11px; text-transform: uppercase;">Unique Installs</th>
          </tr>
        </thead>
        <tbody>
          {history_rows}
        </tbody>
      </table>

      <!-- Direct Links -->
      <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #27272a; text-align: center;">
        <a href="https://pypi.org/project/driftcheck/" style="display: inline-block; margin: 4px; padding: 8px 14px; background: #0284c7; color: #ffffff; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600;">📦 PyPI Package</a>
        <a href="https://pepy.tech/project/driftcheck" style="display: inline-block; margin: 4px; padding: 8px 14px; background: #27272a; color: #e4e4e7; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">📊 PePy Stats</a>
        <a href="https://github.com/BoobalanArjunan/Driftcheck" style="display: inline-block; margin: 4px; padding: 8px 14px; background: #27272a; color: #e4e4e7; text-decoration: none; border-radius: 6px; font-size: 12px; font-weight: 600; border: 1px solid #3f3f46;">💻 GitHub</a>
      </div>

    </div>

    <!-- Footer -->
    <div style="background: #141416; padding: 16px; text-align: center; border-top: 1px solid #27272a; font-size: 11px; color: #71717a;">
      Sent automatically via GitHub Actions Daily Analytics Cron &bull; Driftcheck Project
    </div>

  </div>
</body>
</html>
"""
    return html


def send_email_resend(html_content: str, stats: dict):
    if not RESEND_API_KEY:
        print("[INFO] RESEND_API_KEY not configured. Running in preview mode.")
        print(f"[INFO] Would send to: {TARGET_EMAIL}")
        print(f"[INFO] Stats summary: Total {stats['total_with_mirrors']}, Last week {stats['last_week']}")
        return True

    payload = {
        "from": FROM_EMAIL,
        "to": [TARGET_EMAIL],
        "subject": f"📊 Driftcheck Daily Stats: {stats['total_with_mirrors']} Total Downloads (+{stats['last_day']} yesterday)",
        "html": html_content,
    }

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Driftcheck-Reporter",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_data = json.loads(resp.read().decode())
            print(f"[SUCCESS] Email dispatched successfully! Resend ID: {resp_data.get('id')}")
            return True
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        print(f"[ERROR] Resend API error ({e.code}): {err_body}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}", file=sys.stderr)
        return False


def main():
    print("=== Driftcheck Daily Analytics Reporter ===")
    stats = fetch_stats()
    html = build_html_email(stats)
    success = send_email_resend(html, stats)
    if not success and RESEND_API_KEY:
        sys.exit(1)


if __name__ == "__main__":
    main()

