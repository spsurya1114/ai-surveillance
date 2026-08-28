import os
import json
import uuid
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class RiskEngine:
    def __init__(self, config_path="config.json"):
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load config in RiskEngine: {e}")
        
        self.weights = self.config.get("risk_weights", {
            "loitering": 25,
            "intrusion": 40,
            "wrong_direction": 20,
            "sudden_movement": 20,
            "crowd_anomaly": 15,
            "abandoned_object": 30
        })
        self.window = self.config.get("risk_combination_window", 30.0)
        # person_id -> list of active anomalies: {"anomaly_type": type, "timestamp": timestamp}
        self.profiles = {}

    def update_profile(self, person_id, anomaly, timestamp):
        """Adds an anomaly to a person's profile and returns the combined risk score/level."""
        if not person_id:
            # If no person ID (e.g., crowd anomaly), calculate standalone risk
            atype = anomaly.get("anomaly_type")
            score = self.weights.get(atype, 20)
            return score, self.classify_level(score)

        if person_id not in self.profiles:
            self.profiles[person_id] = []

        # Remove expired anomalies outside the sliding window
        self.profiles[person_id] = [
            a for a in self.profiles[person_id]
            if timestamp - a["timestamp"] <= self.window
        ]

        # Add or update current anomaly
        existing = next((a for a in self.profiles[person_id] if a["anomaly_type"] == anomaly["anomaly_type"]), None)
        if existing:
            existing["timestamp"] = timestamp
        else:
            self.profiles[person_id].append({
                "anomaly_type": anomaly["anomaly_type"],
                "timestamp": timestamp
            })

        # Calculate combined score
        combined_score = sum(self.weights.get(a["anomaly_type"], 20) for a in self.profiles[person_id])
        combined_score = min(100, max(0, combined_score))

        return combined_score, self.classify_level(combined_score)

    def classify_level(self, score):
        if score >= 90:
            return "CRITICAL"
        elif score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        return "LOW"


class AlertManager:
    def __init__(self, config_path="config.json"):
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load config in AlertManager: {e}")

        self.threshold_level = self.config.get("alert_threshold", "HIGH")
        self.cooldown = self.config.get("alert_cooldown", 60.0)
        self.save_path = "data/logs/alerts.json"
        
        # (person_id/object_id, anomaly_type, camera_id) -> last_alert_time
        self.cooldowns = {}
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def should_alert(self, anomaly, risk_score, risk_level, timestamp, update_cooldown=True):
        """Determines if an anomaly should trigger an alert based on risk level and cooldowns."""
        level_hierarchy = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        target_val = level_hierarchy.get(self.threshold_level, 2)
        curr_val = level_hierarchy.get(risk_level, 0)

        # Check threshold
        is_significant = curr_val >= target_val or risk_level == "CRITICAL"
        if not is_significant:
            return False

        # Check cooldown
        pid = anomaly.get("person_id") or anomaly.get("object_id") or "global"
        atype = anomaly["anomaly_type"]
        cam = anomaly["camera_id"]
        key = (pid, atype, cam)

        if key in self.cooldowns:
            if timestamp - self.cooldowns[key] < self.cooldown:
                return False

        if update_cooldown:
            self.cooldowns[key] = timestamp
        return True

    def process_alert(self, anomaly, risk_score, risk_level, timestamp):
        """Processes and routes the alert to dashboard log, local files, and email."""
        if not self.should_alert(anomaly, risk_score, risk_level, timestamp, update_cooldown=True):
            return None

        # Build alert record
        alert_id = str(uuid.uuid4())
        alert_record = {
            "alert_id": alert_id,
            "timestamp": timestamp,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "camera_id": anomaly["camera_id"],
            "person_id": anomaly.get("person_id"),
            "object_id": anomaly.get("object_id"),
            "anomaly_type": anomaly["anomaly_type"],
            "description": anomaly["description"],
            "risk_score": risk_score,
            "risk_level": risk_level,
            "evidence_path": anomaly.get("evidence_path")
        }

        # 1. Log to local alerts file
        self.log_alert_local(alert_record)

        # 2. Email alert
        self.send_email_alert(alert_record)

        return alert_record

    def log_alert_local(self, alert_record):
        alerts = []
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    alerts = json.load(f)
            except:
                alerts = []

        alerts.append(alert_record)

        try:
            with open(self.save_path, "w") as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            print(f"Failed to log alert locally: {e}")

    def send_email_alert(self, alert_record):
        # Read credentials from env
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT")
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        to_email = os.getenv("ALERT_TO_EMAIL")

        # Fallback to config if not in env (though env is preferred for secrets)
        email_config = self.config.get("email_settings", {})
        smtp_server = smtp_server or email_config.get("server")
        smtp_port = smtp_port or email_config.get("port")
        smtp_user = smtp_user or email_config.get("user")
        smtp_pass = smtp_pass or email_config.get("password")
        to_email = to_email or email_config.get("to_email")

        if not (smtp_server and smtp_user and smtp_pass and to_email):
            # SMTP config not available, skip email alert silently
            return

        subject = f"⚠️ [{alert_record['risk_level']}] Security Incident on {alert_record['camera_id']}"
        body = f"""
        Security Alert Triggered:
        -------------------------
        Alert ID: {alert_record['alert_id']}
        Time: {alert_record['timestamp']} seconds
        Camera: {alert_record['camera_id']}
        Anomaly: {alert_record['anomaly_type']}
        Risk Score: {alert_record['risk_score']} ({alert_record['risk_level']})
        Description: {alert_record['description']}
        Evidence: {alert_record.get('evidence_path') or 'None'}
        """

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_email

        try:
            with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(smtp_user, [to_email], msg.as_string())
            print(f"📧 Alert email sent successfully to {to_email}")
        except Exception as e:
            print(f"Failed to send email alert: {e}")
