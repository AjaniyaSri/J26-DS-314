"""Unit tests for C1 inference service, station flood risk catalog, and FastAPI endpoints."""
from __future__ import annotations
import sys
import unittest
import warnings
from pathlib import Path

# Suppress Starlette deprecation warning about httpx vs httpx2 before importing TestClient
warnings.filterwarnings(
    "ignore",
    message=".*httpx.*httpx2.*",
    category=DeprecationWarning,
)

from fastapi.testclient import TestClient  # noqa: E402

C1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C1 / "src"))

from improvements.stations import load_station_catalog, classify_flood_risk
from improvements.inference import C1Predictor
from api.main import app


class TestInferenceAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_station_catalog(C1)
        cls.predictor = C1Predictor(root=C1)
        cls.client = TestClient(app)

    def test_station_catalog_loaded(self):
        self.assertGreaterEqual(len(self.catalog), 39)
        self.assertIn("ellagawa", self.catalog)
        ellagawa = self.catalog["ellagawa"]
        self.assertEqual(ellagawa["basin"], "Kalu Ganga")
        self.assertAlmostEqual(ellagawa["alert_threshold_m"], 10.0)
        self.assertAlmostEqual(ellagawa["minor_flood_threshold_m"], 10.7)
        self.assertAlmostEqual(ellagawa["major_flood_threshold_m"], 12.2)

    def test_flood_risk_classification(self):
        st = self.catalog["ellagawa"]
        # Normal
        res_normal = classify_flood_risk(5.0, st)
        self.assertEqual(res_normal["status"], "NORMAL")
        self.assertEqual(res_normal["severity"], 0)

        # Alert (10.0 <= stage < 10.7)
        res_alert = classify_flood_risk(10.2, st)
        self.assertEqual(res_alert["status"], "ALERT")
        self.assertEqual(res_alert["severity"], 1)

        # Minor Flood (10.7 <= stage < 12.2)
        res_minor = classify_flood_risk(11.0, st)
        self.assertEqual(res_minor["status"], "MINOR_FLOOD")
        self.assertEqual(res_minor["severity"], 2)

        # Major Flood (>= 12.2)
        res_major = classify_flood_risk(12.5, st)
        self.assertEqual(res_major["status"], "MAJOR_FLOOD")
        self.assertEqual(res_major["severity"], 3)

    def test_predictor_single_station(self):
        res = self.predictor.predict_for_record("ellagawa", "2025-12-14")
        self.assertEqual(res["station_id"], "ellagawa")
        self.assertEqual(res["forecast_date"], "2025-12-14")
        self.assertIn("predicted_next_day_stage_m", res)
        self.assertIn("uncertainty_interval_95", res)
        self.assertIn("flood_risk", res)
        self.assertIn("satellite_provenance", res)
        # Check delta identity
        expected_stage = round(res["stage_at_issue_0900_m"] + res["predicted_delta_m"], 3)
        self.assertAlmostEqual(res["predicted_next_day_stage_m"], expected_stage, places=3)

    def test_predictor_network_forecasts(self):
        forecasts = self.predictor.get_latest_network_forecasts()
        self.assertGreater(len(forecasts), 20)

    def test_api_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertEqual(data["component"], "C1_SATELLITE_DEPTH")

    def test_api_stations_endpoint(self):
        response = self.client.get("/api/v1/stations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["count"], 39)

    def test_api_predict_station_endpoint(self):
        response = self.client.get("/api/v1/predict/ellagawa?forecast_date=2025-12-14")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["station_id"], "ellagawa")
        self.assertIn("predicted_next_day_stage_m", data)

    def test_api_custom_predict_endpoint(self):
        payload = {
            "station_id": "ellagawa",
            "stage_at_issue_m": 4.5,
            "stage_lag1_m": 4.2,
            "stage_lag2_m": 4.0,
            "sar_latest_age_hours": 48.0,
            "swi005_d1": 55.0,
            "swi010_d1": 50.0
        }
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["station_id"], "ellagawa")
        self.assertIn("predicted_next_day_stage_m", data)


if __name__ == "__main__":
    unittest.main()
