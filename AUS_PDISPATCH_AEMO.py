# -*- coding: utf-8 -*-
"""
AEMO Predispatch downloader, extractor, plotter, and email sender
- Finds the latest ZIP on NEMWeb Predispatch_Reports
- Extracts CSV(s) to local inputs folder
- Plots daily average RRP & intraday RRP per region
- Emails a simple HTML table + attached plot

Author: Ben
"""

import os
import sys
import time
import math
import shutil
import zipfile
import logging
import datetime as dt
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib


# ------------------------- CONFIG -------------------------
BASE_DIR = Path(r"C:\Users\benja\OneDrive\Bureau\Dynasty_analysis\AEMO")
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "outputs"
TMP_DIR = BASE_DIR / "tmp"
PLOT_PATH = OUTPUTS_DIR / "RRP_per_PERIODID_plot.png"

NEMWEB_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/Predispatch_Reports/"

REGIONS = ["QLD1", "NSW1", "VIC1", "SA1"]


FROM_EMAIL = "test@dynasty.com"
TO_EMAIL = [
    "benjamin.vaillant@yahoo.fr"
]
CC_EMAIL = ""
SMTP_SERVER = "mailhost.prod.dynasty.com"
SMTP_PORT = 25

# Main loop control
CHECK_INTERVAL_SEC = 30
OPERATIONAL_WINDOW = (10, 20)  # 10:00–19:59 local


# ---------------------- SETUP LOGGING ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("aemo_predispatch")


# ---------------------- UTILITIES -------------------------
def ensure_dirs():
    for d in [INPUTS_DIR, OUTPUTS_DIR, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (AEMO-Fetch; +https://example.local)"
        }
    )
    return s


def list_zip_links(session, url):
    """Return list of (href, absolute_url, text) for .zip links found."""
    r = session.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    zips = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".zip"):
            abs_url = urljoin(url, href)
            zips.append((href, abs_url, a.get_text(strip=True)))
    return zips


def pick_latest_zip(zips):
    """
    Best-effort selection of the 'latest':
    - If names contain timestamps, max by name; else pick last in listing order.
    """
    if not zips:
        return None
    try:
        # Many NEMWeb files have sortable names; try max by filename
        latest = max(zips, key=lambda t: t[0])
        return latest[1]
    except Exception:
        # Fallback: last in list
        return zips[-1][1]


def download_file(session, url, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path


def extract_csvs(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract only CSV files to dest_dir. Handles nested folders in ZIP."""
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".csv") and not info.is_dir():
                # Flatten nested paths: keep only the base name
                out_path = dest_dir / Path(info.filename).name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(out_path)
    return extracted


# ---------------------- PLOTTING --------------------------
def plot_rrp_daily_and_intraday(csv_path: Path, regions: list[str], save_path: Path) -> dict:
    """
    Reads AEMO CSV (header at row 4), plots:
      1) Daily average RRP per region
      2) Intraday RRP per region
    Returns dict: {region: {date_str: avg_rrp_or_None}}
    """
    # AEMO predispatch CSVs often have 3-line header notes; data starts at line 4
    df = pd.read_csv(csv_path, header=3)

    required_cols = {"REGIONID", "PERIODID", "RRP"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.loc[:, ["REGIONID", "PERIODID", "RRP"]].copy()
    df["PERIODID"] = pd.to_datetime(df["PERIODID"], errors="coerce", utc=False)
    df["RRP"] = pd.to_numeric(df["RRP"], errors="coerce")
    df = df.dropna(subset=["REGIONID", "PERIODID", "RRP"])

    df = df[df["REGIONID"].isin(regions)].copy()
    if df.empty:
        logger.warning(f"No rows for selected regions in {csv_path.name}")
        return {}

    df["DATE"] = df["PERIODID"].dt.date

    grouped = (
        df.groupby(["REGIONID", "DATE"], as_index=False)["RRP"]
        .mean()
        .sort_values(["REGIONID", "DATE"])
    )

    # ---- Plot Daily Averages ----
    plt.figure(figsize=(12, 6))
    for region in regions:
        r = grouped[grouped["REGIONID"] == region]
        if not r.empty:
            plt.plot(r["DATE"], r["RRP"], marker="o", linestyle="-", label=region)
    plt.title("Daily Average RRP by Region")
    plt.xlabel("Date")
    plt.ylabel("Average RRP ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save as a temporary buffer then continue plotting intraday on a new figure
    daily_path = save_path.with_name(save_path.stem + "_daily.png")
    plt.savefig(daily_path, dpi=150)
    plt.close()

    # ---- Plot Intraday (PERIODID) ----
    plt.figure(figsize=(12, 6))
    for region in regions:
        r = df[df["REGIONID"] == region].sort_values("PERIODID")
        if not r.empty:
            plt.plot(r["PERIODID"], r["RRP"], marker="o", linestyle="-", markersize=3, label=region)
    plt.title("RRP per PERIODID by Region")
    plt.xlabel("PERIODID")
    plt.ylabel("RRP ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Build average price map (by available dates in the CSV)
    avg_map: dict[str, dict[str, float | None]] = {}
    for region in regions:
        r = grouped[grouped["REGIONID"] == region]
        region_dates = {pd.to_datetime(d).strftime("%Y-%m-%d"): float(v) for d, v in zip(r["DATE"], r["RRP"])}
        avg_map[region] = region_dates

    logger.info(f"Saved plots: {daily_path} and {save_path}")
    return avg_map


# ---------------------- EMAIL -----------------------------
def send_email_with_image(
    from_email: str,
    to_email: list[str],
    subject: str,
    cc_email: str,
    average_prices: dict,
    image_path: Path,
    smtp_server: str = SMTP_SERVER,
    smtp_port: int = SMTP_PORT,
):
    # Derive date headers from data provided
    all_dates = set()
    for region_dict in average_prices.values():
        all_dates.update(region_dict.keys())
    dates_sorted = sorted(all_dates)

    regions_sorted = sorted(average_prices.keys())

    html = [
        "<html><body>",
        "<p>Please find below the analysis for each region and day:</p>",
        '<table border="1" cellpadding="4" cellspacing="0">',
        "<tr><th>Date</th>",
    ]
    for r in regions_sorted:
        html.append(f"<th>{r}</th>")
    html.append("</tr>")

    for d in dates_sorted:
        html.append(f"<tr><td>{d}</td>")
        for r in regions_sorted:
            val = average_prices.get(r, {}).get(d)
            cell = f"${val:.0f}" if val is not None and not math.isnan(val) else "N/A"
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</table>")
    html.append('<p><img src="cid:image1"></p>')
    html.append("</body></html>")
    html = "".join(html)

    msg = MIMEMultipart("related")
    msg["From"] = from_email
    msg["To"] = ", ".join(to_email)
    msg["Subject"] = subject

    recipients = list(to_email)
    if cc_email:
        msg["Cc"] = cc_email
        # split on comma or semicolon and strip spaces
        cc_list = [x.strip() for x in cc_email.replace(";", ",").split(",") if x.strip()]
        recipients.extend(cc_list)

    msg.attach(MIMEText(html, "html"))

    # attach image
    with open(image_path, "rb") as imgf:
        img = MIMEImage(imgf.read(), name=image_path.name)
        img.add_header("Content-ID", "<image1>")
        msg.attach(img)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.sendmail(from_email, recipients, msg.as_string())
        logger.info("Email sent successfully")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


# ---------------------- MAIN LOOP -------------------------
def within_window(now: dt.datetime, start_h: int, end_h: int) -> bool:
    return start_h <= now.hour < end_h


def main():
    ensure_dirs()
    session = make_session()
    processed_files: set[str] = set()

    while True:
        now = dt.datetime.now()
        if not within_window(now, *OPERATIONAL_WINDOW):
            logger.info("Outside operational window. Exiting.")
            sys.exit(0)

        try:
            zips = list_zip_links(session, NEMWEB_URL)
            if not zips:
                logger.warning("No ZIP links found on the page.")
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            latest_url = pick_latest_zip(zips)
            if not latest_url:
                logger.warning("Could not determine latest ZIP link.")
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            zip_name = Path(urlparse(latest_url).path).name
            if zip_name in processed_files:
                logger.info(f"Already processed {zip_name}. Waiting...")
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            logger.info(f"Latest ZIP: {latest_url}")
            tmp_zip = TMP_DIR / zip_name
            download_file(session, latest_url, tmp_zip)
            extracted_csvs = extract_csvs(tmp_zip, INPUTS_DIR)

            # Clean the temp zip
            try:
                tmp_zip.unlink(missing_ok=True)
            except Exception:
                pass

            if not extracted_csvs:
                logger.warning("No CSVs extracted from the ZIP.")
                processed_files.add(zip_name)
                time.sleep(CHECK_INTERVAL_SEC)
                continue

            # Process all CSVs found (usually one)
            combined_price_map: dict[str, dict[str, float | None]] = {}
            for csv_path in extracted_csvs:
                logger.info(f"Processing {csv_path.name}")
                price_map = plot_rrp_daily_and_intraday(csv_path, REGIONS, PLOT_PATH)
                # merge into combined map
                for region, dmap in price_map.items():
                    combined_price_map.setdefault(region, {}).update(dmap)

            if combined_price_map:
                # Build a concise subject using latest available date if present
                all_dates = set()
                for dmap in combined_price_map.values():
                    all_dates.update(dmap.keys())
                date_for_subject = max(all_dates) if all_dates else now.strftime("%Y-%m-%d")

                # compose a short price summary for subject (using the latest date values)
                parts = []
                for r in REGIONS:
                    v = combined_price_map.get(r, {}).get(date_for_subject)
                    if v is not None and not math.isnan(v):
                        parts.append(f"{r} ${v:.0f}")
                price_snip = " | ".join(parts) if parts else "No prices"

                subject = f"AUS - PDS {date_for_subject} - {price_snip}"
                send_email_with_image(
                    FROM_EMAIL, TO_EMAIL, subject, CC_EMAIL, combined_price_map, PLOT_PATH
                )

            processed_files.add(zip_name)

        except Exception as e:
            logger.exception(f"Error in main cycle: {e}")

        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    main()
