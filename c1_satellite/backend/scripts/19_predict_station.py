#!/usr/bin/env python3
"""Phase 19: Command-line prediction tool for C1 river stage forecasting.

Usage examples:
  python c1_satellite/scripts/19_predict_station.py --station ellagawa
  python c1_satellite/scripts/19_predict_station.py --station ellagawa --date 2025-12-14
  python c1_satellite/scripts/19_predict_station.py --list-stations
  python c1_satellite/scripts/19_predict_station.py --network-overview
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / "src"))

from improvements.inference import C1Predictor


def main():
    parser = argparse.ArgumentParser(description="FloodRisk Sri Lanka C1 — Gauge Prediction Tool")
    parser.add_argument("--station", "-s", type=str, help="Station ID (e.g., ellagawa, badalgama, dunamale)")
    parser.add_argument("--date", "-d", type=str, help="Forecast date (YYYY-MM-DD)")
    parser.add_argument("--model", "-m", type=str, default="A_ENH_2B", help="Model family (A_ENH_2B, STAGE_TEMPORAL)")
    parser.add_argument("--list-stations", action="store_true", help="Print all 39 monitoring stations")
    parser.add_argument("--network-overview", action="store_true", help="Print latest status across national network")
    args = parser.parse_args()

    predictor = C1Predictor(root=C1, model_name=args.model)

    if args.list_stations:
        stations = predictor.get_stations()
        print(f"\n=== SRI LANKA DMC RIVER MONITORING GAUGES ({len(stations)} STATIONS) ===")
        print(f"{'STATION ID':<16} {'NAME':<18} {'BASIN':<18} {'ALERT':<8} {'MINOR':<8} {'MAJOR':<8}")
        print("-" * 80)
        for s in sorted(stations, key=lambda x: x["basin"]):
            alert = f"{s['alert_threshold_m']:.2f}m" if s['alert_threshold_m'] is not None else "N/A"
            minor = f"{s['minor_flood_threshold_m']:.2f}m" if s['minor_flood_threshold_m'] is not None else "N/A"
            major = f"{s['major_flood_threshold_m']:.2f}m" if s['major_flood_threshold_m'] is not None else "N/A"
            print(f"{s['station_id']:<16} {s['station_name']:<18} {s['basin']:<18} {alert:<8} {minor:<8} {major:<8}")
        return

    if args.network_overview:
        forecasts = predictor.get_latest_network_forecasts()
        print(f"\n=== NATIONAL NETWORK FORECAST OVERVIEW ({len(forecasts)} ACTIVE GAUGES) ===")
        print(f"{'STATION':<16} {'DATE':<12} {'NOW (m)':<10} {'PRED (m)':<10} {'DELTA (m)':<10} {'RISK STATUS':<15}")
        print("-" * 75)
        for f in forecasts:
            now = f"{f['stage_at_issue_0900_m']:.2f}"
            pred = f"{f['predicted_next_day_stage_m']:.2f}"
            delta = f"{f['predicted_delta_m']:+.3f}"
            status = f['flood_risk']['predicted_next_day']['status']
            print(f"{f['station_id']:<16} {f['forecast_date']:<12} {now:<10} {pred:<10} {delta:<10} {status:<15}")
        return

    station = args.station or "ellagawa"
    res = predictor.predict_for_record(station, args.date)

    print("\n" + "=" * 65)
    print(f" FLOOD FORECAST REPORT: {res['station_name'].upper()} ({res['basin'].upper()} BASIN)")
    print("=" * 65)
    print(f"Forecast Issued Date  : {res['forecast_date']} 09:30 SLST")
    print(f"Target Forecast Lead  : {res['target_observation_time']}")
    print(f"Active Model Family   : {res['model_family']} (ExtraTrees Regularized Ensemble)")
    print(f"Observed Stage at 0900: {res['stage_at_issue_0900_m']:.3f} m")
    print(f"Predicted Stage Delta : {res['predicted_delta_m']:+.4f} m")
    print(f"PREDICTED 24h STAGE   : {res['predicted_next_day_stage_m']:.3f} m")
    print(f"95% Confidence Bounds : [{res['uncertainty_interval_95']['lower_m']:.3f} m to {res['uncertainty_interval_95']['upper_m']:.3f} m]")
    print("-" * 65)
    print(f"Current Risk Level    : {res['flood_risk']['current']['status']} ({res['flood_risk']['current']['message']})")
    print(f"Predicted Risk Level  : {res['flood_risk']['predicted_next_day']['status']} ({res['flood_risk']['predicted_next_day']['message']})")
    if res['flood_risk']['risk_escalating']:
        print("  WARNING: Water level is projected to escalate into higher flood threshold!")
    print("-" * 65)
    print("SATELLITE EXPLANATORY SIGNALS:")
    sar_age = f"{res['satellite_provenance']['sar_latest_age_hours']:.1f} hours" if res['satellite_provenance']['sar_latest_age_hours'] else "N/A"
    swi_now = f"{res['satellite_provenance']['swi005_latest']:.1f}%" if res['satellite_provenance']['swi005_latest'] else "N/A"
    trend = f"{res['satellite_provenance']['swi005_7day_trend']:+.2f}%/day" if res['satellite_provenance']['swi005_7day_trend'] else "N/A"
    print(f"  Sentinel-1 SAR Age  : {sar_age}")
    print(f"  ASCAT SWI005 Soil   : {swi_now} (7-Day Wetness Trend: {trend})")

    if "validation_ground_truth" in res:
        gt = res["validation_ground_truth"]
        print("-" * 65)
        print("HISTORICAL VALIDATION COMPARISON:")
        print(f"  Actual Measured Next-Day Stage : {gt['actual_next_day_stage_m']:.3f} m")
        print(f"  Actual Change Observed         : {gt['actual_delta_m']:+.4f} m")
        print(f"  Absolute Prediction Error      : {gt['absolute_error_m']:.4f} m ({gt['absolute_error_m']*100:.1f} cm)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise SystemExit(f"PREDICT STOP: {e}")
