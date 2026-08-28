import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import json
import shutil
from anomaly.detector import AnomalyDetector
from anomaly.alert_manager import RiskEngine, AlertManager
from rag.temporal_reasoning import TemporalRetriever

class TestSurveillanceSystem(unittest.TestCase):
    def setUp(self):
        # Create temp config for testing
        self.test_config = {
            "loitering_threshold": 2.0,
            "restricted_zones": {
                "TEST_CAM": [
                    {
                        "name": "zone_restricted",
                        "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]
                    }
                ]
            },
            "allowed_directions": {
                "TEST_CAM": {
                    "angle": 90,  # moving down (+y)
                    "tolerance": 45
                }
            },
            "velocity_threshold": 50.0,
            "crowd_threshold": 2,
            "abandoned_object_threshold": 2.0,
            "risk_weights": {
                "loitering": 25,
                "intrusion": 40,
                "wrong_direction": 20,
                "sudden_movement": 20,
                "crowd_anomaly": 15,
                "abandoned_object": 30
            },
            "risk_combination_window": 10.0,
            "alert_threshold": "LOW",  # Changed to LOW for cooldown test visibility
            "alert_cooldown": 5.0,
            "tracking": {
                "max_missing_frames": 3
            }
        }
        with open("test_config.json", "w") as f:
            json.dump(self.test_config, f)
            
        # Initialize detector with test config
        self.detector = AnomalyDetector(config_path="test_config.json")
        self.risk_engine = RiskEngine(config_path="test_config.json")
        self.alert_manager = AlertManager(config_path="test_config.json")
        
        # Prepare a clean temp logs directory for temporal testing
        self.temp_logs_dir = "data/temp_test_logs"
        if os.path.exists(self.temp_logs_dir):
            shutil.rmtree(self.temp_logs_dir)
        os.makedirs(self.temp_logs_dir, exist_ok=True)
        self.retriever = TemporalRetriever(config_path="test_config.json", logs_dir=self.temp_logs_dir)

    def tearDown(self):
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")
        if os.path.exists(self.temp_logs_dir):
            shutil.rmtree(self.temp_logs_dir)
            
    def test_loitering_threshold(self):
        # Frame 0: person enters outside restricted zone
        dets = [{"id": 1, "class": 0, "bbox": [150, 150, 160, 160]}]
        anoms = self.detector.update(frame_id=0, fps=10, detections=dets, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 0)
        
        # Frame 10 (1 second later): under threshold (2.0s)
        anoms = self.detector.update(frame_id=10, fps=10, detections=dets, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 0)
        
        # Frame 30 (3 seconds later): exceeds threshold
        anoms = self.detector.update(frame_id=30, fps=10, detections=dets, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0]["anomaly_type"], "loitering")

    def test_restricted_zone_detection(self):
        # Outside restricted zone but close to boundary
        dets = [{"id": 1, "class": 0, "bbox": [110, 110, 120, 120]}]
        anoms = self.detector.update(frame_id=0, fps=10, detections=dets, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 0)
        
        # Inside restricted zone (center is (50, 50))
        # Set frame_id to 19 (dt=1.9s) so velocity < 50 px/s and duration < 2.0s loitering threshold
        dets_inside = [{"id": 1, "class": 0, "bbox": [40, 40, 60, 60]}]
        anoms = self.detector.update(frame_id=19, fps=10, detections=dets_inside, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0]["anomaly_type"], "intrusion")
        self.assertEqual(anoms[0]["zone"], "zone_restricted")

    def test_direction_detection(self):
        # Moving downwards (allowed: 90 deg, which is +y)
        # Bounding box x-coordinates placed outside restricted zone (x=200)
        for fid, y in enumerate([50, 60, 70, 80, 95]):
            dets = [{"id": 1, "class": 0, "bbox": [190, y-10, 210, y+10]}]
            anoms = self.detector.update(frame_id=fid, fps=10, detections=dets, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 0)  # moving in allowed direction
        
        # Moving upwards (-y, angle = 270 deg) - violates allowed direction 90 deg
        self.detector = AnomalyDetector(config_path="test_config.json")  # reset
        for fid, y in enumerate([100, 90, 80, 70, 50]):
            dets = [{"id": 2, "class": 0, "bbox": [190, y-10, 210, y+10]}]
            anoms = self.detector.update(frame_id=fid, fps=10, detections=dets, camera_id="TEST_CAM")
        # The last update should trigger wrong direction
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0]["anomaly_type"], "wrong_direction")

    def test_velocity_detection(self):
        # Slow movement (placed outside restricted zone)
        dets1 = [{"id": 1, "class": 0, "bbox": [190, 190, 210, 210]}]
        dets2 = [{"id": 1, "class": 0, "bbox": [192, 192, 212, 212]}]
        self.detector.update(frame_id=0, fps=10, detections=dets1, camera_id="TEST_CAM")
        anoms = self.detector.update(frame_id=1, fps=10, detections=dets2, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 0)
        
        # Sudden movement (dist = ~141 pixels, dt = 0.1s, velocity = ~1410 px/s > threshold 50.0)
        dets_fast = [{"id": 1, "class": 0, "bbox": [290, 290, 310, 310]}]
        anoms = self.detector.update(frame_id=2, fps=10, detections=dets_fast, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0]["anomaly_type"], "sudden_movement")

    def test_crowd_threshold(self):
        # 1 person (under threshold 2, placed outside restricted zone)
        dets = [{"id": 1, "class": 0, "bbox": [190, 190, 210, 210]}]
        anoms = self.detector.update(frame_id=0, fps=10, detections=dets, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 0)
        
        # 3 people (exceeds threshold 2)
        dets_crowd = [
            {"id": 1, "class": 0, "bbox": [190, 190, 210, 210]},
            {"id": 2, "class": 0, "bbox": [230, 230, 250, 250]},
            {"id": 3, "class": 0, "bbox": [270, 270, 290, 290]}
        ]
        anoms = self.detector.update(frame_id=1, fps=10, detections=dets_crowd, camera_id="TEST_CAM")
        self.assertEqual(len(anoms), 1)
        self.assertEqual(anoms[0]["anomaly_type"], "crowd_anomaly")

    def test_risk_score_calculation_and_classification(self):
        # Test loitering risk scoring
        score, level = self.risk_engine.update_profile("P1", {"anomaly_type": "loitering"}, 1.0)
        self.assertEqual(score, 25)
        self.assertEqual(level, "LOW")
        
        # Test combined risk scoring within window (loitering + intrusion = 25 + 40 = 65)
        score, level = self.risk_engine.update_profile("P1", {"anomaly_type": "intrusion"}, 2.0)
        self.assertEqual(score, 65)
        self.assertEqual(level, "MEDIUM")
        
        # Test clamping and high level (loitering + intrusion + abandoned_object = 25 + 40 + 30 = 95)
        score, level = self.risk_engine.update_profile("P1", {"anomaly_type": "abandoned_object"}, 3.0)
        self.assertEqual(score, 95)
        self.assertEqual(level, "CRITICAL")

    def test_alert_cooldown_and_deduplication(self):
        anomaly = {
            "camera_id": "TEST_CAM",
            "anomaly_type": "intrusion",
            "description": "Intrusion test",
            "timestamp": 1.0,
            "person_id": "P1"
        }
        
        # First alert: should trigger
        self.assertTrue(self.alert_manager.should_alert(anomaly, 40, "MEDIUM", 1.0))
        # Log it to cache time
        self.alert_manager.process_alert(anomaly, 40, "MEDIUM", 1.0)
        
        # Immediate subsequent alert: should be throttled by cooldown (5.0s)
        self.assertFalse(self.alert_manager.should_alert(anomaly, 40, "MEDIUM", 2.0))
        
        # Alert after cooldown: should trigger
        self.assertTrue(self.alert_manager.should_alert(anomaly, 40, "MEDIUM", 7.0))

    def test_temporal_event_retrieving_and_grounding(self):
        # Create a mock json event log file
        mock_events = [
            {
                "event_id": "evt-01",
                "timestamp": 5.0,
                "camera_id": "TEST_CAM",
                "person_id": "1",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "left",
                "description": "Person 1 entered zone left"
            },
            {
                "event_id": "evt-02",
                "timestamp": 15.0,
                "camera_id": "TEST_CAM",
                "person_id": "1",
                "event_type": "intrusion",
                "severity": "HIGH",
                "risk_score": 40,
                "zone": "zone_restricted",
                "description": "Person 1 intruded restricted zone"
            }
        ]
        
        log_file = os.path.join(self.temp_logs_dir, "TEST_CAM.json")
        with open(log_file, "w") as f:
            json.dump(mock_events, f)
            
        # Test temporal query retrieval
        retrieved_events = self.retriever.load_all_events()
        self.assertEqual(len(retrieved_events), 2)
        
        # Test filtering by person_id
        filters = self.retriever.parse_query_filters("What did person 1 do?")
        filtered = self.retriever.filter_events(retrieved_events, filters)
        self.assertEqual(len(filtered), 2)
        
        # Test filtering by start/end time
        # Camera starts at 10:00:00, so event 1 is at 10:00:05, event 2 at 10:00:15
        filters_time = self.retriever.parse_query_filters("What happened between 10:00:00 and 10:00:10?")
        filtered_time = self.retriever.filter_events(retrieved_events, filters_time)
        self.assertEqual(len(filtered_time), 1)
        self.assertEqual(filtered_time[0]["event_id"], "evt-01")

    def test_llm_grounding_insufficient_evidence(self):
        # Empty logs in directory
        retrieved_events = self.retriever.load_all_events()
        self.assertEqual(len(retrieved_events), 0)
        
        # Query for person 999
        answer, context = self.retriever.query("What did P999 do?")
        self.assertIn("retrieved evidence", answer)
        self.assertIn("Insufficient evidence", context)

    def test_event_logger_track_loss_grace_period(self):
        from tracking.logger import EventLogger
        
        logger_path = os.path.join(self.temp_logs_dir, "test_grace.json")
        event_logger = EventLogger(save_path=logger_path, config_path="test_config.json")
        
        # a. Person enters -> one ENTRY event
        dets = [{"id": 1, "class": 0, "bbox": [150, 150, 160, 160]}]
        events = event_logger.update(frame_id=0, fps=10, detections=dets, frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "entry")
        self.assertEqual(events[0]["person_id"], 1)
        
        # b. Person disappears for fewer than max_missing_frames (3 frames) -> NO EXIT event
        # Frame 1: absent (1st missing frame)
        events = event_logger.update(frame_id=1, fps=10, detections=[], frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 0)
        
        # Frame 2: absent (2nd missing frame)
        events = event_logger.update(frame_id=2, fps=10, detections=[], frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 0)
        
        # c. Person reappears within grace period -> NO new ENTRY event
        # Frame 3: reappears
        events = event_logger.update(frame_id=3, fps=10, detections=dets, frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 0)
        
        # d. Person disappears for more than max_missing_frames (3 frames) -> ONE EXIT event
        # Frame 4: absent (1st missing frame)
        events = event_logger.update(frame_id=4, fps=10, detections=[], frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 0)
        
        # Frame 5: absent (2nd missing frame)
        events = event_logger.update(frame_id=5, fps=10, detections=[], frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 0)
        
        # Frame 6: absent (3rd missing frame)
        events = event_logger.update(frame_id=6, fps=10, detections=[], frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 0)
        
        # Frame 7: absent (4th missing frame, exceeds max_missing_frames=3) -> EXIT event should be generated
        events = event_logger.update(frame_id=7, fps=10, detections=[], frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "exit")
        self.assertEqual(events[0]["person_id"], 1)
        
        # e. Person reappears after confirmed exit -> ONE new ENTRY event
        # Frame 8: reappears
        events = event_logger.update(frame_id=8, fps=10, detections=dets, frame_width=640, camera_id="TEST_CAM")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "entry")
        self.assertEqual(events[0]["person_id"], 1)

    def test_anomaly_logging_and_alerting_decoupling(self):
        from tracking.logger import EventLogger
        from anomaly.alert_manager import RiskEngine, AlertManager
        
        logger_path = os.path.join(self.temp_logs_dir, "test_decouple.json")
        event_logger = EventLogger(save_path=logger_path, config_path="test_config.json")
        
        # Setup RiskEngine and AlertManager with config threshold = "HIGH"
        custom_config = self.test_config.copy()
        custom_config["alert_threshold"] = "HIGH"
        custom_config["alert_cooldown"] = 10.0
        
        with open("test_decouple_config.json", "w") as f:
            json.dump(custom_config, f)
            
        risk_engine = RiskEngine(config_path="test_decouple_config.json")
        alert_manager = AlertManager(config_path="test_decouple_config.json")
        
        try:
            # Anomaly metadata
            anomaly_low = {
                "camera_id": "TEST_CAM",
                "anomaly_type": "loitering",  # weight 25 -> LOW risk
                "description": "Loitering anomaly",
                "timestamp": 1.0,
                "person_id": "P1"
            }
            
            anomaly_high = {
                "camera_id": "TEST_CAM",
                "anomaly_type": "intrusion",  # weight 40 -> combined 25+40=65 (medium). Let's make it intrusion + wrong_direction = 85 (HIGH)
                "description": "Intrusion anomaly",
                "timestamp": 2.0,
                "person_id": "P1"
            }
            
            # --- Test case a & b: Low-risk anomaly -> logged but no alert
            score_1, level_1 = risk_engine.update_profile("P1", anomaly_low, 1.0)
            self.assertEqual(level_1, "LOW")
            
            # Check if alert manager should alert (should be False because alert_threshold is HIGH)
            is_alert_1 = alert_manager.should_alert(anomaly_low, score_1, level_1, 1.0, update_cooldown=False)
            self.assertFalse(is_alert_1)  # Low risk, should not alert
            
            # Log the anomaly (should be logged)
            logged_1 = event_logger.log_anomaly(anomaly_low, score_1, level_1)
            self.assertEqual(logged_1["event_type"], "anomaly")
            self.assertEqual(logged_1["anomaly_type"], "loitering")
            self.assertEqual(logged_1["severity"], "LOW")
            self.assertIsNone(logged_1["evidence_path"])
            
            # --- Test case c: High-risk anomaly -> logged AND alert generated
            # Add intrusion (40) and wrong_direction (20) to reach 85 (HIGH)
            risk_engine.update_profile("P1", anomaly_high, 2.0)
            anomaly_wd = {
                "camera_id": "TEST_CAM",
                "anomaly_type": "wrong_direction",
                "description": "Wrong direction anomaly",
                "timestamp": 2.0,
                "person_id": "P1"
            }
            score_2, level_2 = risk_engine.update_profile("P1", anomaly_wd, 2.0)
            self.assertEqual(level_2, "HIGH")
            
            # Check if alert manager should alert (should be True since level is HIGH)
            is_alert_2 = alert_manager.should_alert(anomaly_wd, score_2, level_2, 2.0, update_cooldown=False)
            self.assertTrue(is_alert_2)
            
            # Save evidence mock paths only for the alert
            evidence_img = "/path/to/img.jpg"
            evidence_clip = "/path/to/clip.mp4"
            
            logged_2 = event_logger.log_anomaly(anomaly_wd, score_2, level_2, evidence_path=evidence_img, evidence_clip_path=evidence_clip)
            self.assertEqual(logged_2["event_type"], "anomaly")
            self.assertEqual(logged_2["severity"], "HIGH")
            self.assertEqual(logged_2["evidence_path"], evidence_img)
            self.assertEqual(logged_2["evidence_clip_path"], evidence_clip)
            
            # Process alert (simulating alert dispatch)
            alert_rec = alert_manager.process_alert(logged_2, score_2, level_2, 2.0)
            self.assertIsNotNone(alert_rec)
            self.assertEqual(alert_rec["risk_level"], "HIGH")
            
            # --- Test case d: Alert cooldown prevents duplicate alerts
            # Immediate repeat of same high-risk anomaly type at t=3.0s (within cooldown of 10.0s)
            is_alert_3 = alert_manager.should_alert(anomaly_wd, score_2, level_2, 3.0, update_cooldown=False)
            self.assertFalse(is_alert_3)  # Cooldown blocks it
            
            # --- Test case e: Evidence is generated only for actual alerts
            # Since is_alert_3 is False, we do not generate evidence paths:
            logged_3 = event_logger.log_anomaly(anomaly_wd, score_2, level_2, evidence_path=None, evidence_clip_path=None)
            self.assertIsNone(logged_3["evidence_path"])
            self.assertIsNone(logged_3["evidence_clip_path"])

        finally:
            if os.path.exists("test_decouple_config.json"):
                os.remove("test_decouple_config.json")

    def test_risk_accumulation_and_decay(self):
        from anomaly.alert_manager import RiskEngine, AlertManager
        
        # Setup test configuration
        custom_config = self.test_config.copy()
        custom_config["alert_threshold"] = "HIGH"
        custom_config["risk_combination_window"] = 30.0  # 30-second combination window
        
        with open("test_risk_config.json", "w") as f:
            json.dump(custom_config, f)
            
        risk_engine = RiskEngine(config_path="test_risk_config.json")
        alert_manager = AlertManager(config_path="test_risk_config.json")
        
        try:
            # Person 1 triggers intrusion (40) at t=1.0s
            score, level = risk_engine.update_profile("P1", {"anomaly_type": "intrusion"}, 1.0)
            self.assertEqual(score, 40)
            self.assertEqual(level, "MEDIUM")
            
            # Person 1 triggers loitering (25) at t=15.0s (within 30s window)
            score, level = risk_engine.update_profile("P1", {"anomaly_type": "loitering"}, 15.0)
            self.assertEqual(score, 65)
            self.assertEqual(level, "MEDIUM")
            
            # Person 1 triggers sudden_movement (20) at t=25.0s (within 30s window of all)
            score, level = risk_engine.update_profile("P1", {"anomaly_type": "sudden_movement"}, 25.0)
            self.assertEqual(score, 85)
            self.assertEqual(level, "HIGH")
            
            # Verify alert eligibility (threshold is HIGH)
            is_alert = alert_manager.should_alert({"anomaly_type": "sudden_movement", "camera_id": "TEST_CAM"}, score, level, 25.0, update_cooldown=False)
            self.assertTrue(is_alert)
            
            # At t=35.0s, another event occurs. intrusion at t=1.0s decays (35 - 1 = 34 > 30s)
            score, level = risk_engine.update_profile("P1", {"anomaly_type": "loitering"}, 35.0) # refresh loitering
            # Score should be loitering (25) + sudden_movement (20) = 45
            self.assertEqual(score, 45)
            self.assertEqual(level, "MEDIUM")
            
            # Verify alert eligibility at t=35.0s (now below HIGH threshold)
            is_alert = alert_manager.should_alert({"anomaly_type": "loitering", "camera_id": "TEST_CAM"}, score, level, 35.0, update_cooldown=False)
            self.assertFalse(is_alert)

        finally:
            if os.path.exists("test_risk_config.json"):
                os.remove("test_risk_config.json")

    def test_temporal_rag_and_llm(self):
        from rag.temporal_reasoning import TemporalRetriever
        import rag.llm as llm
        
        # 1. Setup a set of surveillance events in our temp logs directory
        temp_log_file = os.path.join(self.temp_logs_dir, "CAM_01.json")
        mock_events = [
            {
                "event_id": "evt-1",
                "timestamp": 1.0,
                "camera_id": "CAM_01",
                "person_id": 1,
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "confidence": 1.0,
                "zone": "center",
                "description": "Person 1 entered scene at zone center."
            },
            {
                "event_id": "evt-2",
                "timestamp": 3.0,
                "camera_id": "CAM_01",
                "person_id": 1,
                "event_type": "anomaly",
                "anomaly_type": "intrusion",
                "severity": "MEDIUM",
                "risk_score": 40,
                "confidence": 0.95,
                "zone": "restricted_A",
                "description": "Person 1 intruded restricted zone 'restricted_A'."
            },
            {
                "event_id": "evt-3",
                "timestamp": 6.0,
                "camera_id": "CAM_01",
                "person_id": 1,
                "event_type": "anomaly",
                "anomaly_type": "loitering",
                "severity": "MEDIUM",
                "risk_score": 65,
                "confidence": 0.98,
                "zone": "center",
                "description": "Person 1 loitering for 5.0 seconds."
            },
            {
                "event_id": "evt-4",
                "timestamp": 15.0,
                "camera_id": "CAM_01",
                "person_id": 2,
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "confidence": 1.0,
                "zone": "left",
                "description": "Person 2 entered scene."
            },
            {
                "event_id": "evt-5",
                "timestamp": 25.0,
                "camera_id": "CAM_01",
                "person_id": 2,
                "event_type": "anomaly",
                "anomaly_type": "sudden_movement",
                "severity": "HIGH",
                "risk_score": 75,
                "confidence": 0.90,
                "zone": "left",
                "description": "Person 2 moved with sudden velocity."
            }
        ]
        
        with open(temp_log_file, "w") as f:
            json.dump(mock_events, f)
            
        retriever = TemporalRetriever(config_path="test_config.json", logs_dir=self.temp_logs_dir)
        
        # 2. Test chronological sorting using event timestamps
        loaded = retriever.load_all_events()
        self.assertEqual(len(loaded), 5)
        self.assertEqual([e["event_id"] for e in loaded], ["evt-1", "evt-2", "evt-3", "evt-4", "evt-5"])
        
        # 3. Test filtering combinations
        f1 = retriever.parse_query_filters("Show events in CAM 1")
        self.assertEqual(f1["camera_id"], "CAM_01")
        
        f2 = retriever.parse_query_filters("Timeline for Person 1")
        self.assertEqual(f2["person_id"], "1")
        
        f3 = retriever.parse_query_filters("Who did intrusion?")
        self.assertEqual(f3["anomaly_type"], "intrusion")
        self.assertEqual(f3["event_type"], "anomaly")
        
        f4 = retriever.parse_query_filters("Show medium risk events")
        self.assertEqual(f4["risk_level"], "MEDIUM")
        
        f5 = retriever.parse_query_filters("Events with risk above 50")
        self.assertEqual(f5["min_risk_score"], 50)
        
        f6 = retriever.parse_query_filters("What happened between 5 and 10 seconds?")
        self.assertEqual(f6["start_time"], "10:00:05")
        self.assertEqual(f6["end_time"], "10:00:10")
        
        # 4. Test Query/Context Retrieval sets for 5 realistic NL queries
        
        # Query A: "What happened to Person 1 between 5 and 10 seconds?"
        ans_a, ctx_a = retriever.query("What happened to Person 1 between 5 and 10 seconds?")
        self.assertIn("evt-3", ctx_a)
        self.assertNotIn("evt-1", ctx_a)
        self.assertNotIn("evt-4", ctx_a)
        
        # Query B: "What anomalies occurred in CAM 1?"
        ans_b, ctx_b = retriever.query("What anomalies occurred in CAM 1?")
        self.assertIn("evt-2", ctx_b)
        self.assertIn("evt-3", ctx_b)
        self.assertIn("evt-5", ctx_b)
        self.assertNotIn("evt-1", ctx_b)
        self.assertNotIn("evt-4", ctx_b)
        
        # Query C: "Which person had the highest risk score?"
        ans_c, ctx_c = retriever.query("Which person had the highest risk score?")
        self.assertIn("Person 2", ans_c)
        self.assertIn("75", ans_c)
        
        # Query D: "Were there any high-risk events?"
        ans_d, ctx_d = retriever.query("Were there any high-risk events?")
        self.assertIn("Yes", ans_d)
        
        # Query E: "What happened to Person 99 between 50 and 60 seconds?" (Empty Result Handling)
        ans_f, ctx_f = retriever.query("What happened to Person 99 between 50 and 60 seconds?")
        self.assertEqual(ans_f, "I don't have enough retrieved evidence to determine that.")
        self.assertEqual(ctx_f, "Empty context: no matching events found in existing logs.")
        
        # 5. Verify GROQ_API_KEY available vs unavailable mock response generation
        original_client = llm.client
        llm.client = None  # Force offline fallback
        
        fallback_ans = llm.ask_llm("Timeline for Person 1", ctx_b)
        self.assertIn("Timeline for person 1", fallback_ans)
        
        llm.client = original_client

    def test_concise_rag_queries(self):
        # Prepare a clean set of logs for these tests
        custom_log_file = os.path.join(self.temp_logs_dir, "CAM_02.json")
        
        mock_events = [
            {
                "event_id": "e-1",
                "timestamp": 1.0,
                "camera_id": "CAM_02",
                "person_id": "P1",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "center",
                "description": "P1 entry"
            },
            {
                "event_id": "e-2",
                "timestamp": 3.0,
                "camera_id": "CAM_02",
                "person_id": "P1",
                "event_type": "anomaly",
                "anomaly_type": "loitering",
                "severity": "LOW",
                "risk_score": 25,
                "zone": "center",
                "description": "P1 loitered"
            },
            {
                "event_id": "e-3",
                "timestamp": 5.0,
                "camera_id": "CAM_02",
                "person_id": "P2",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "center",
                "description": "P2 entry"
            },
            {
                "event_id": "e-4",
                "timestamp": 6.0,
                "camera_id": "CAM_02",
                "person_id": "P2",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "center",
                "description": "P2 entry duplicate"
            },
            {
                "event_id": "e-5",
                "timestamp": 7.0,
                "camera_id": "CAM_02",
                "person_id": "P3",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "center",
                "description": "P3 entry"
            },
            {
                "event_id": "e-6",
                "timestamp": 9.0,
                "camera_id": "CAM_02",
                "person_id": "P3",
                "event_type": "anomaly",
                "anomaly_type": "intrusion",
                "severity": "MEDIUM",
                "risk_score": 40,
                "zone": "zone_restricted",
                "description": "P3 intrusion"
            },
            {
                "event_id": "e-7",
                "timestamp": 10.0,
                "camera_id": "CAM_02",
                "person_id": "P4",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "center",
                "description": "P4 entry"
            },
            {
                "event_id": "e-8",
                "timestamp": 12.0,
                "camera_id": "CAM_02",
                "person_id": "P5",
                "event_type": "entry",
                "severity": "LOW",
                "risk_score": 0,
                "zone": "center",
                "description": "P5 entry"
            },
            {
                "event_id": "e-9",
                "timestamp": 15.0,
                "camera_id": "CAM_02",
                "person_id": "P5",
                "event_type": "anomaly",
                "anomaly_type": "wrong_direction",
                "severity": "MEDIUM",
                "risk_score": 65,
                "zone": "center",
                "description": "P5 wrong direction"
            }
        ]
        
        with open(custom_log_file, "w") as f:
            json.dump(mock_events, f)
            
        retriever = TemporalRetriever(config_path="test_config.json", logs_dir=self.temp_logs_dir)
        
        # Test 1 & 6: Person count
        ans_count, ctx_count = retriever.query("How many persons are there in the video?")
        self.assertIn("5 unique persons", ans_count)
        self.assertNotIn("e-1", ans_count)
        
        # Test 2: Person list
        ans_list, _ = retriever.query("Who are the persons detected?")
        self.assertIn("P1", ans_list)
        self.assertIn("P2", ans_list)
        self.assertIn("P3", ans_list)
        self.assertIn("P4", ans_list)
        self.assertIn("P5", ans_list)
        self.assertNotIn("e-1", ans_list)
        
        # Test 3: Anomaly count
        ans_anom_count, _ = retriever.query("How many anomalies occurred?")
        self.assertIn("3 anomalies", ans_anom_count)
        self.assertIn("Loitering: 1", ans_anom_count)
        self.assertIn("Intrusion: 1", ans_anom_count)
        self.assertIn("Wrong-direction: 1", ans_anom_count)
        
        ans_specific_count, _ = retriever.query("How many intrusion events occurred?")
        self.assertIn("1 intrusion anomalies were detected", ans_specific_count)

        # Test 4: Highest risk
        ans_risk, _ = retriever.query("Which person had the highest risk?")
        self.assertIn("P5", ans_risk)
        self.assertIn("65", ans_risk)
        self.assertIn("MEDIUM", ans_risk)
        
        # Test 7: Temporal query
        ans_temp, ctx_temp = retriever.query("What happened between 10:05:00 and 10:05:04?")
        self.assertIn("10:05:01", ctx_temp)
        self.assertIn("10:05:03", ctx_temp)
        self.assertNotIn("10:05:05", ctx_temp)
        
        # Test 8: Dashboard answer (raw grounding records are not included in the normal answer)
        self.assertNotIn("event_id", ans_count)
        self.assertNotIn("description", ans_count)
        self.assertNotIn("timestamp", ans_count)

        # Test filter_evidence_by_query directly
        # A. Person count: should return exactly 5 representative events (one per unique person P1, P2, P3, P4, P5)
        filtered_pcount = retriever.filter_evidence_by_query("How many persons are there in the video?", mock_events)
        self.assertEqual(len(filtered_pcount), 5)
        self.assertEqual(set(e["person_id"] for e in filtered_pcount), {"P1", "P2", "P3", "P4", "P5"})

        # B. Person specific: should return only P1 events (2 events: entry and loitering)
        filtered_p1 = retriever.filter_evidence_by_query("What did P1 do?", mock_events)
        self.assertEqual(len(filtered_p1), 2)
        self.assertTrue(all(e["person_id"] == "P1" for e in filtered_p1))

        # C. Anomaly count/list: should return only the 3 anomalies (loitering, intrusion, wrong direction)
        filtered_anoms = retriever.filter_evidence_by_query("What anomalies occurred?", mock_events)
        self.assertEqual(len(filtered_anoms), 3)
        self.assertTrue(all(e.get("anomaly_type") is not None for e in filtered_anoms))

        # D. Intrusion yes/no: should return only intrusion anomaly events (1 event: P3 intrusion)
        filtered_intrusion = retriever.filter_evidence_by_query("Did anyone enter the restricted zone?", mock_events)
        self.assertEqual(len(filtered_intrusion), 1)
        self.assertEqual(filtered_intrusion[0]["anomaly_type"], "intrusion")

        # E. Highest risk: should return only events of P5 (highest risk score 65)
        filtered_risk = retriever.filter_evidence_by_query("Which person had the highest risk?", mock_events)
        self.assertEqual(len(filtered_risk), 2)
        self.assertTrue(all(e["person_id"] == "P5" for e in filtered_risk))
        
        # Test 5: Empty events
        os.remove(custom_log_file)
        cam1_file = os.path.join(self.temp_logs_dir, "CAM_01.json")
        if os.path.exists(cam1_file):
            os.remove(cam1_file)
        test_cam_file = os.path.join(self.temp_logs_dir, "TEST_CAM.json")
        if os.path.exists(test_cam_file):
            os.remove(test_cam_file)
            
        retriever_empty = TemporalRetriever(config_path="test_config.json", logs_dir=self.temp_logs_dir)
        ans_empty, _ = retriever_empty.query("How many persons are there?")
        self.assertEqual(ans_empty, "No persons were detected in the available event records.")

if __name__ == "__main__":
    unittest.main()
