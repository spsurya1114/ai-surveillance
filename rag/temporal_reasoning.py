import os
import json
import re
from datetime import datetime, timedelta
from rag.llm import ask_llm

class TemporalRetriever:
    def __init__(self, config_path="config.json", logs_dir="data/logs"):
        self.logs_dir = logs_dir
        self.config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load config in TemporalRetriever: {e}")

        self.camera_start_times = self.config.get("camera_start_times", {
            "CAM_01": "2026-08-23T10:00:00",
            "CAM_02": "2026-08-23T10:05:00",
            "CAM_03": "2026-08-23T10:10:00"
        })

    def get_absolute_time(self, camera_id, relative_seconds):
        """Converts relative video seconds to an absolute datetime string."""
        start_str = self.camera_start_times.get(camera_id, "2026-08-23T10:00:00")
        try:
            start_dt = datetime.fromisoformat(start_str)
        except:
            start_dt = datetime(2026, 8, 23, 10, 0, 0)
        
        abs_dt = start_dt + timedelta(seconds=relative_seconds)
        return abs_dt.strftime("%H:%M:%S")

    def time_str_to_seconds(self, time_str):
        """Converts HH:MM:SS or HH:MM string to seconds from start of day."""
        parts = time_str.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 3600 + int(parts[1]) * 60
        except:
            pass
        return None

    def load_all_events(self):
        """Loads all events from all camera log files."""
        all_events = []
        if not os.path.exists(self.logs_dir):
            return all_events

        for filename in os.listdir(self.logs_dir):
            if filename.endswith(".json") and filename != "alerts.json":
                path = os.path.join(self.logs_dir, filename)
                try:
                    with open(path, "r") as f:
                        events = json.load(f)
                        # Inject absolute time into each event
                        cam_id = filename.replace(".json", "")
                        for e in events:
                            e["camera_id"] = e.get("camera_id", cam_id)
                            e["abs_time"] = self.get_absolute_time(e["camera_id"], e["timestamp"])
                        all_events.extend(events)
                except Exception as e:
                    print(f"Error loading events from {path}: {e}")

        # Sort chronologically by absolute time
        all_events.sort(key=lambda x: (x.get("date", "2026-08-23"), x.get("abs_time", "00:00:00"), x["timestamp"]))
        return all_events

    def parse_query_filters(self, query):
        """Extracts structured filters from the natural language query."""
        query_lower = query.lower()
        filters = {
            "person_id": None,
            "camera_id": None,
            "anomaly_type": None,
            "event_type": None,
            "risk_level": None,
            "min_risk_score": None,
            "start_time": None,
            "end_time": None
        }

        # 1. Extract Person ID
        person_match = re.search(r'\b(?:p|person|pid)\s*[-_]?\s*0*(\d+)\b', query_lower)
        if person_match:
            filters["person_id"] = person_match.group(1)

        # 2. Extract Camera ID
        camera_match = re.search(r'\b(?:cam|camera)\s*[-_]?\s*0*(\d+)\b', query_lower)
        if camera_match:
            filters["camera_id"] = f"CAM_0{camera_match.group(1)}"

        # 3. Extract Anomaly Type
        anomaly_keywords = {
            "loitering": ["loiter", "loitering"],
            "intrusion": ["intrusion", "intrude", "restricted", "zone"],
            "wrong_direction": ["wrong", "direction", "allowed direction"],
            "sudden_movement": ["sudden", "running", "velocity", "speed"],
            "crowd_anomaly": ["crowd", "people count", "crowded"],
            "abandoned_object": ["abandoned", "bag", "backpack", "suitcase", "object"]
        }
        for atype, keywords in anomaly_keywords.items():
            if any(k in query_lower for k in keywords):
                filters["anomaly_type"] = atype
                break

        # 4. Extract Event Type
        if "anomaly" in query_lower or "anomalies" in query_lower or "suspicious" in query_lower or "incident" in query_lower:
            filters["event_type"] = "anomaly"
        elif "entry" in query_lower or "entered" in query_lower or "enter" in query_lower:
            if not ("restricted" in query_lower or "intrusion" in query_lower or "zone" in query_lower):
                filters["event_type"] = "entry"
        elif "exit" in query_lower or "exited" in query_lower:
            filters["event_type"] = "exit"
        elif "movement" in query_lower or "move" in query_lower or "moved" in query_lower:
            filters["event_type"] = "movement"

        if filters["anomaly_type"]:
            filters["event_type"] = "anomaly"

        # 5. Extract Risk Level
        for rlevel in ["low", "medium", "high", "critical"]:
            if re.search(r'\b' + rlevel + r'\b', query_lower):
                filters["risk_level"] = rlevel.upper()
                break

        # 6. Extract Min Risk Score
        score_match = re.search(r'\b(?:risk|score)\s*(?:of|above|>=|>)?\s*(\d{1,3})\b', query_lower)
        if score_match:
            filters["min_risk_score"] = int(score_match.group(1))

        # 7. Extract Start & End Time
        # A. Check for seconds range: e.g. "between 5 and 10 seconds" or "5 to 10 seconds"
        secs_match = re.search(r'\b(?:between|from)\s*(\d+)\s*(?:and|to)\s*(\d+)\s*(?:seconds|second|s|sec)\b', query_lower)
        if not secs_match:
            secs_match = re.search(r'\b(\d+)\s*(?:to|-)\s*(\d+)\s*(?:seconds|second|s|sec)\b', query_lower)
            
        if secs_match:
            sec_start = int(secs_match.group(1))
            sec_end = int(secs_match.group(2))
            base_secs = 10 * 3600  # Default camera start time 10:00:00
            
            start_total = base_secs + sec_start
            h_start = (start_total // 3600) % 24
            m_start = (start_total % 3600) // 60
            s_start = start_total % 60
            filters["start_time"] = f"{h_start:02d}:{m_start:02d}:{s_start:02d}"
            
            end_total = base_secs + sec_end
            h_end = (end_total // 3600) % 24
            m_end = (end_total % 3600) // 60
            s_end = end_total % 60
            filters["end_time"] = f"{h_end:02d}:{m_end:02d}:{s_end:02d}"
        else:
            # Check for single seconds limit: e.g. "after 5 seconds", "before 10 seconds"
            single_sec_after = re.search(r'\b(?:after|from)\s*(\d+)\s*(?:seconds|second|s|sec)\b', query_lower)
            single_sec_before = re.search(r'\b(?:before|to)\s*(\d+)\s*(?:seconds|second|s|sec)\b', query_lower)
            base_secs = 10 * 3600
            
            if single_sec_after:
                sec_val = int(single_sec_after.group(1))
                start_total = base_secs + sec_val
                h = (start_total // 3600) % 24
                m = (start_total % 3600) // 60
                s = start_total % 60
                filters["start_time"] = f"{h:02d}:{m:02d}:{s:02d}"
            elif single_sec_before:
                sec_val = int(single_sec_before.group(1))
                end_total = base_secs + sec_val
                h = (end_total // 3600) % 24
                m = (end_total % 3600) // 60
                s = end_total % 60
                filters["end_time"] = f"{h:02d}:{m:02d}:{s:02d}"
            else:
                # B. Match HH:MM:SS or HH:MM formats
                times = re.findall(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', query_lower)
                if times:
                    if "between" in query_lower and len(times) >= 2:
                        filters["start_time"] = times[0]
                        filters["end_time"] = times[1]
                    elif "after" in query_lower or "from" in query_lower:
                        filters["start_time"] = times[0]
                    elif "before" in query_lower or "to" in query_lower:
                        filters["end_time"] = times[0]
                    else:
                        filters["start_time"] = times[0]

                # C. Handle 12-hour AM/PM formats
                am_pm_match = re.findall(r'\b(\d{1,2})\s*(am|pm)\b', query_lower)
                if am_pm_match and not times:
                    parsed_times = []
                    for hour, period in am_pm_match:
                        h = int(hour)
                        if period == "pm" and h < 12:
                            h += 12
                        elif period == "am" and h == 12:
                            h = 0
                        parsed_times.append(f"{h:02d}:00:00")
                    
                    if len(parsed_times) >= 2 and "between" in query_lower:
                        filters["start_time"] = parsed_times[0]
                        filters["end_time"] = parsed_times[1]
                    elif "after" in query_lower:
                        filters["start_time"] = parsed_times[0]
                    elif "before" in query_lower:
                        filters["end_time"] = parsed_times[0]
                    else:
                        filters["start_time"] = parsed_times[0]

        return filters

    def filter_events(self, events, filters):
        """Applies filters to the chronological event list."""
        filtered = []
        for e in events:
            # Person ID Match
            if filters["person_id"]:
                pid_str = str(e.get("person_id", "")).strip().lower().lstrip("p").lstrip("0")
                if pid_str != filters["person_id"].lstrip("0"):
                    continue

            # Camera ID Match
            if filters["camera_id"] and e.get("camera_id") != filters["camera_id"]:
                continue

            # Event Type Match
            if filters["event_type"]:
                e_type = e.get("event_type")
                if filters["event_type"] == "anomaly":
                    # Anomaly filter matches if event_type is 'anomaly', OR if it is a legacy anomaly type
                    if e_type not in ["anomaly", "loitering", "intrusion", "wrong_direction", "sudden_movement", "crowd_anomaly", "abandoned_object"]:
                        continue
                else:
                    if e_type != filters["event_type"]:
                        continue

            # Anomaly Type Match
            if filters["anomaly_type"]:
                e_type = e.get("event_type")
                a_type = e.get("anomaly_type")
                if e_type != filters["anomaly_type"] and a_type != filters["anomaly_type"]:
                    # Backwards compatible mapping if they request 'loitering'
                    if filters["anomaly_type"] == "loitering" and e_type != "loitering" and a_type != "loitering":
                        continue
                    elif filters["anomaly_type"] != "loitering":
                        continue

            # Risk Level Match
            if filters["risk_level"]:
                level_hierarchy = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
                e_level = e.get("severity", "LOW")
                if level_hierarchy.get(e_level, 0) < level_hierarchy.get(filters["risk_level"], 0):
                    continue

            # Min Risk Score Match
            if filters["min_risk_score"] and e.get("risk_score", 0) < filters["min_risk_score"]:
                continue

            # Time Range Match
            abs_time = e.get("abs_time")
            if abs_time:
                e_secs = self.time_str_to_seconds(abs_time)
                if filters["start_time"]:
                    start_secs = self.time_str_to_seconds(filters["start_time"])
                    if start_secs and e_secs < start_secs:
                        continue
                if filters["end_time"]:
                    end_secs = self.time_str_to_seconds(filters["end_time"])
                    if end_secs and e_secs > end_secs:
                        continue

            filtered.append(e)
        return filtered

    def get_grounded_timeline_context(self, filtered_events):
        """Converts filtered events to natural sentences for LLM grounding."""
        if not filtered_events:
            return "Empty context: no matching events found in existing logs."

        sentences = []
        for e in filtered_events:
            abs_time = e.get("abs_time", "00:00:00")
            cam = e["camera_id"]
            etype = e.get("anomaly_type") if e.get("event_type") == "anomaly" else e["event_type"]
            pid = e.get("person_id")
            oid = e.get("object_id")
            risk = e.get("risk_score", 0)
            level = e.get("severity", "LOW")
            eid = e.get("event_id", "N/A")

            # Entry / Movement / Exit
            if etype == "entry":
                sentences.append(f"[{eid}] At {abs_time}, person P{pid} entered zone '{e.get('zone')}' in camera {cam} (Risk: {risk}, Severity: {level}).")
            elif etype == "movement":
                sentences.append(f"[{eid}] At {abs_time}, person P{pid} moved from '{e.get('from_zone')}' to '{e.get('to_zone')}' in camera {cam} (Risk: {risk}, Severity: {level}).")
            elif etype == "exit":
                sentences.append(f"[{eid}] At {abs_time}, person P{pid} exited scene from zone '{e.get('zone')}' in camera {cam} (Risk: {risk}, Severity: {level}).")
            # Anomalies
            else:
                target_label = f"person P{pid}" if pid else f"object ID {oid}"
                sentences.append(f"[{eid}] At {abs_time}, anomaly '{etype}' associated with {target_label} detected in camera {cam} - Description: {e['description']} (Risk: {risk}, Severity: {level}).")

        return "\n".join(sentences)

    def _extract_unique_persons(self, events):
        pids = []
        for e in events:
            pid = e.get("person_id")
            if pid is not None and pid != "":
                # normalize pid: e.g. "P001" or "P1" or 1 -> "1"
                pid_str = str(pid).strip().lower().lstrip("p").lstrip("0")
                if pid_str:
                    try:
                        pids.append(int(pid_str))
                    except ValueError:
                        pids.append(pid_str)
        return sorted(list(set(pids)))

    def _format_person_id(self, pid):
        if isinstance(pid, int):
            return f"P{pid}"
        return f"P{pid}".upper()

    def _format_list(self, items):
        if not items:
            return ""
        if len(items) == 1:
            return str(items[0])
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(str(x) for x in items[:-1]) + f", and {items[-1]}"

    def _get_risk_level(self, score):
        if score <= 39:
            return "LOW"
        elif score <= 69:
            return "MEDIUM"
        elif score <= 89:
            return "HIGH"
        else:
            return "CRITICAL"

    def classify_intent(self, query):
        query_lower = query.lower()
        
        # 1. PERSON_COUNT
        if re.search(r'how\s+many\s+(?:person|people|individual|id|ids|user|users)', query_lower) or \
           re.search(r'count\s+(?:of\s+)?(?:person|people|individual|id|ids|user|users)', query_lower):
            return "PERSON_COUNT"
            
        # 2. PERSON_LIST
        if re.search(r'\b(?:who|list)\b.*\b(?:person|people|individual|id|ids|user|users)\b', query_lower) or \
           re.search(r'\b(?:who is|who are|who was|who were)\b', query_lower):
            return "PERSON_LIST"
            
        # 3. HIGHEST_RISK
        if "highest risk" in query_lower or "max risk" in query_lower or "maximum risk" in query_lower or "highest score" in query_lower or "highest recorded risk" in query_lower:
            return "HIGHEST_RISK"
            
        # 4. PERSON_TIMELINE
        if "timeline" in query_lower or "history" in query_lower or \
           re.search(r'what\s+(?:did|happened\s+to|was\s+done\s+by)\s+(?:person|p\d*)\b', query_lower):
            return "PERSON_TIMELINE"
            
        # 5. ANOMALY_COUNT
        if re.search(r'how\s+many\s+(?:anomaly|anomalies|incident|incidents|intrusion|loitering|wrong|direction|sudden|movement|running|crowd|abandoned)', query_lower) or \
           re.search(r'count\s+(?:of\s+)?(?:anomaly|anomalies|incident|incidents|intrusion|loitering|wrong|direction|sudden|movement|running|crowd|abandoned)', query_lower):
            return "ANOMALY_COUNT"

        # 6. ANOMALY_LIST
        if re.search(r'\b(?:what|list)\s+(?:anomaly|anomalies|incident|incidents)\b', query_lower):
            return "ANOMALY_LIST"

        # 7. YES_NO_EVENT
        if re.search(r'\b(?:did|was|were|is|are|has|have)\b.*\b(?:anyone|someone|any|intrusion|loitering|intrude|wrong|direction|sudden|running|crowd|abandoned)\b', query_lower):
            return "YES_NO_EVENT"

        # 8. CAMERA_SUMMARY
        if "camera" in query_lower and ("most" in query_lower or "highest" in query_lower or "summary" in query_lower or "active" in query_lower):
            return "CAMERA_SUMMARY"

        # 9. TIME_RANGE_SUMMARY
        if "around" in query_lower or "between" in query_lower or "time range" in query_lower or "at" in query_lower and re.search(r'\b\d{1,2}:\d{2}\b', query_lower):
            return "TIME_RANGE_SUMMARY"

        # 10. EVENT_LOOKUP
        if "event" in query_lower and re.search(r'[0-9a-f]{8}-[0-9a-f]{4}', query_lower):
            return "EVENT_LOOKUP"

        return "GENERAL_SURVEILLANCE_QUERY"

    def process_deterministic_query(self, intent, filtered_events, query):
        query_lower = query.lower()
        
        ANOMALY_NAMES = {
            "loitering": "loitering",
            "intrusion": "intrusion",
            "wrong_direction": "wrong-direction",
            "sudden_movement": "sudden movement",
            "crowd_anomaly": "crowd anomaly",
            "abandoned_object": "abandoned object"
        }

        # PERSON_COUNT
        if intent == "PERSON_COUNT":
            pids = self._extract_unique_persons(filtered_events)
            if not pids:
                return "No persons were detected in the available event records."
            count = len(pids)
            return f"{count} unique persons were detected in the video."

        # PERSON_LIST
        elif intent == "PERSON_LIST":
            pids = self._extract_unique_persons(filtered_events)
            if not pids:
                return "No persons were detected in the available event records."
            count = len(pids)
            persons_str = self._format_list([self._format_person_id(pid) for pid in pids])
            return f"{count} unique persons were detected: {persons_str}."

        # HIGHEST_RISK
        elif intent == "HIGHEST_RISK":
            max_score = -1
            person_max_scores = {}
            pids_raw = {}
            for e in filtered_events:
                pid = e.get("person_id")
                score = e.get("risk_score")
                if pid is not None and pid != "" and score is not None:
                    pid_str = str(pid).strip().lower().lstrip("p").lstrip("0")
                    if pid_str:
                        try:
                            pid_val = int(pid_str)
                        except ValueError:
                            pid_val = pid_str
                        
                        pids_raw[pid_val] = pid
                        person_max_scores[pid_val] = max(person_max_scores.get(pid_val, 0), int(score))
                        if int(score) > max_score:
                            max_score = int(score)
            
            if not person_max_scores or max_score == -1:
                return "No risk score records were found for any person in the available event records."
            
            tied_persons = [pid for pid, score in person_max_scores.items() if score == max_score]
            severity = self._get_risk_level(max_score)
            
            if len(tied_persons) == 1:
                pid_formatted = self._format_person_id(tied_persons[0])
                raw_id = pids_raw[tied_persons[0]]
                return f"Person {pid_formatted} (Person {raw_id}) reached the highest recorded risk score of {max_score} ({severity})."
            else:
                persons_formatted_list = [self._format_person_id(pid) for pid in sorted(tied_persons)]
                persons_str = self._format_list(persons_formatted_list)
                return f"Persons {persons_str} reached the highest recorded risk score of {max_score} ({severity})."

        # PERSON_TIMELINE
        elif intent == "PERSON_TIMELINE":
            # Extract target person ID
            person_match = re.search(r'\b(?:p|person|pid)\s*[-_]?\s*0*(\d+)\b', query_lower)
            target_pid = person_match.group(1) if person_match else None
            
            if not target_pid:
                filters = self.parse_query_filters(query)
                if filters.get("person_id"):
                    target_pid = filters["person_id"]
            
            person_events = []
            for e in filtered_events:
                pid = e.get("person_id")
                if pid is not None and pid != "":
                    pid_str = str(pid).strip().lower().lstrip("p").lstrip("0")
                    if target_pid and pid_str == target_pid.lstrip("0"):
                        person_events.append(e)
            
            if not person_events:
                display_pid = f"P{target_pid}" if target_pid else "specified"
                return f"No events found for Person {display_pid} in the available event records."
            
            person_events.sort(key=lambda x: (x.get("date", "2026-08-23"), x.get("abs_time", "00:00:00"), x["timestamp"]))
            
            lines = [f"Person P{target_pid}:"]
            for e in person_events:
                time_str = e.get("abs_time", e.get("timestamp"))
                etype = e.get("event_type")
                
                if etype == "entry":
                    zone = e.get("zone", "scene")
                    action = f"Entered {zone}."
                elif etype == "exit":
                    zone = e.get("zone", "scene")
                    action = "Exited the scene." if zone == "scene" else f"Exited from {zone}."
                elif etype == "movement":
                    fz = e.get("from_zone", "unknown")
                    tz = e.get("to_zone", "unknown")
                    action = f"Moved from {fz} to {tz}."
                elif etype == "anomaly" or etype in ANOMALY_NAMES:
                    atype = e.get("anomaly_type") or etype
                    if atype == "intrusion":
                        zone_str = f" in zone '{e.get('zone')}'" if e.get("zone") else ""
                        action = f"Restricted-zone intrusion{zone_str}."
                    elif atype == "loitering":
                        action = "Loitering detected."
                    elif atype == "wrong_direction":
                        action = "Wrong-direction movement."
                    elif atype == "sudden_movement":
                        action = "Sudden/running movement."
                    elif atype == "crowd_anomaly":
                        action = "Crowd anomaly."
                    elif atype == "abandoned_object":
                        action = "Abandoned object detected."
                    else:
                        action = f"{atype.replace('_', ' ').capitalize()} anomaly."
                else:
                    desc = e.get("description", "")
                    desc_clean = re.sub(r'^(Person\s*\d+|P\d+)\s+', '', desc, flags=re.IGNORECASE)
                    action = desc_clean.capitalize() if desc_clean else "Active in scene."
                
                lines.append(f"- {time_str} — {action}")
            
            return "\n".join(lines)

        # ANOMALY_COUNT
        elif intent == "ANOMALY_COUNT":
            anomaly_keywords = {
                "loitering": ["loiter", "loitering"],
                "intrusion": ["intrusion", "intrude", "restricted", "zone"],
                "wrong_direction": ["wrong", "direction", "allowed direction"],
                "sudden_movement": ["sudden", "running", "velocity", "speed"],
                "crowd_anomaly": ["crowd", "people count", "crowded"],
                "abandoned_object": ["abandoned", "bag", "backpack", "suitcase", "object"]
            }
            target_atype = None
            for atype, keywords in anomaly_keywords.items():
                if any(k in query_lower for k in keywords):
                    target_atype = atype
                    break
            
            if target_atype:
                count = sum(1 for e in filtered_events if e.get("anomaly_type") == target_atype or e.get("event_type") == target_atype)
                name = ANOMALY_NAMES.get(target_atype, target_atype).replace("_", " ")
                if count == 0:
                    return f"No {name} anomalies were detected in the available event records."
                return f"{count} {name} anomalies were detected."
            else:
                anomaly_counts = {}
                total_anomalies = 0
                for e in filtered_events:
                    etype = e.get("event_type")
                    if etype == "anomaly" or etype in ANOMALY_NAMES:
                        atype = e.get("anomaly_type") or etype
                        anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
                        total_anomalies += 1
                
                if total_anomalies == 0:
                    return "No anomalies were detected in the available event records."
                
                breakdown_lines = []
                for atype in sorted(anomaly_counts.keys()):
                    atype_count = anomaly_counts[atype]
                    name = ANOMALY_NAMES.get(atype, atype).replace("_", " ").capitalize()
                    breakdown_lines.append(f"- {name}: {atype_count}")
                breakdown_str = "\n".join(breakdown_lines)
                return f"{total_anomalies} anomalies were logged:\n{breakdown_str}"

        # ANOMALY_LIST
        elif intent == "ANOMALY_LIST":
            anomaly_list = []
            for e in filtered_events:
                etype = e.get("event_type")
                if etype == "anomaly" or etype in ANOMALY_NAMES:
                    atype = e.get("anomaly_type") or etype
                    name = ANOMALY_NAMES.get(atype, atype).replace("_", " ").capitalize()
                    pid = f"P{e.get('person_id')}" if e.get('person_id') else f"object ID {e.get('object_id', 'N/A')}"
                    time_str = e.get("abs_time", e.get("timestamp"))
                    anomaly_list.append(f"- {time_str} — {name} involving {pid} in camera {e['camera_id']}.")
            
            if not anomaly_list:
                return "No anomalies were detected in the available event records."
            return "The following anomalies were detected:\n" + "\n".join(anomaly_list)

        # YES_NO_EVENT
        elif intent == "YES_NO_EVENT":
            event_name = "Activity"
            if "intrusion" in query_lower or "restricted" in query_lower:
                event_name = "Restricted-zone intrusion"
            elif "loitering" in query_lower:
                event_name = "Loitering"
            elif "wrong" in query_lower or "direction" in query_lower:
                event_name = "Wrong-direction movement"
            elif "sudden" in query_lower or "running" in query_lower:
                event_name = "Sudden/running movement"
            elif "crowd" in query_lower:
                event_name = "Crowd anomaly"
            elif "abandoned" in query_lower:
                event_name = "Abandoned object"
            elif "entry" in query_lower or "entered" in query_lower:
                event_name = "Entry"
            elif "exit" in query_lower or "exited" in query_lower:
                event_name = "Exit"
            elif "movement" in query_lower or "move" in query_lower:
                event_name = "Movement"
                
            if filtered_events:
                ans = f"Yes. {event_name} was detected."
                pids = self._extract_unique_persons(filtered_events)
                if pids:
                    persons_str = self._format_list([self._format_person_id(pid) for pid in pids])
                    ans += f"\nPersons involved: {persons_str}."
                return ans
            else:
                return f"No {event_name.lower()} was detected in the available event records."

        # CAMERA_SUMMARY
        elif intent == "CAMERA_SUMMARY":
            cam_counts = {}
            for e in filtered_events:
                if e.get("event_type") == "anomaly" or e.get("event_type") in ANOMALY_NAMES:
                    cam = e.get("camera_id")
                    if cam:
                        cam_counts[cam] = cam_counts.get(cam, 0) + 1
            if not cam_counts:
                return "No camera records containing anomalies were found in the available event records."
            most_active_cam = max(cam_counts, key=cam_counts.get)
            return f"Camera {most_active_cam} registered the highest count of anomalies with a total of {cam_counts[most_active_cam]} events."

        # EVENT_LOOKUP
        elif intent == "EVENT_LOOKUP":
            if filtered_events:
                e = filtered_events[0]
                time_str = e.get("abs_time", e.get("timestamp"))
                pid = f"P{e.get('person_id')}" if e.get('person_id') else f"object ID {e.get('object_id', 'N/A')}"
                return f"Event {e.get('event_id')} details:\n- Time: {time_str}\n- Camera: {e['camera_id']}\n- Type: {e.get('event_type')}\n- Description: {e['description']}"
            else:
                return "Event not found."

        return "Query processing failed."

    def retrieve_events(self, user_question):
        """Retrieves and filters events based on the question's filters."""
        filters = self.parse_query_filters(user_question)
        events = self.load_all_events()
        return self.filter_events(events, filters)

    def query(self, user_question):
        """Main RAG/LLM grounded query pipeline."""
        # 1. Retrieve and filter events
        filtered_events = self.retrieve_events(user_question)
        self.last_filtered_events = filtered_events

        # Load all events to check if database is empty
        events = self.load_all_events()

        # 4. Classify query intent
        intent = self.classify_intent(user_question)
        print("Classified intent:", intent)
        self.last_query_intent = intent

        # 5. Format grounding context (kept internally for display in dashboard expander)
        if not events:
            context = "Insufficient evidence: no event records were found matching your query."
        else:
            context = self.get_grounded_timeline_context(filtered_events)

        # If no events at all:
        if not events:
            if intent in ["PERSON_COUNT", "PERSON_LIST"]:
                answer = "No persons were detected in the available event records."
            elif intent in ["ANOMALY_COUNT", "ANOMALY_LIST"]:
                answer = "No anomalies were detected in the available event records."
            else:
                answer = "I don't have enough retrieved evidence to determine that."
            return answer, context

        # 6. Check if we can solve it deterministically
        deterministic_intents = {
            "PERSON_COUNT", "PERSON_LIST", "HIGHEST_RISK",
            "ANOMALY_COUNT", "EVENT_LOOKUP"
        }

        # If filtered_events is empty:
        if not filtered_events:
            if intent in ["PERSON_COUNT", "PERSON_LIST"]:
                answer = "No persons were detected in the available event records."
            elif intent in ["ANOMALY_COUNT", "ANOMALY_LIST"]:
                answer = "No anomalies were detected in the available event records."
            elif intent == "YES_NO_EVENT":
                answer = self.process_deterministic_query(intent, filtered_events, user_question)
            elif intent == "CAMERA_SUMMARY":
                answer = self.process_deterministic_query(intent, filtered_events, user_question)
            elif intent == "EVENT_LOOKUP":
                answer = "Event not found."
            else:
                answer = "I don't have enough retrieved evidence to determine that."
            return answer, context

        if intent in deterministic_intents:
            answer = self.process_deterministic_query(intent, filtered_events, user_question)
        else:
            # Fall back to LLM (or simulator) for summaries and general reasoning
            answer = ask_llm(user_question, context)

        return answer, context

    def filter_evidence_by_query(self, question, retrieved_events):
        """Filters retrieved events to keep only evidence relevant to the question/intent."""
        intent = self.classify_intent(question)
        query_lower = question.lower()
        
        # Helper to normalize pid
        def normalize_pid(pid):
            if pid is None or pid == "":
                return ""
            return str(pid).strip().lower().lstrip("p").lstrip("0")
            
        if not retrieved_events:
            return []

        # 1. PERSON_COUNT / PERSON_LIST
        if intent in ["PERSON_COUNT", "PERSON_LIST"]:
            seen_persons = set()
            filtered = []
            for e in retrieved_events:
                pid = e.get("person_id")
                if pid is not None and pid != "":
                    pid_norm = normalize_pid(pid)
                    if pid_norm not in seen_persons:
                        seen_persons.add(pid_norm)
                        filtered.append(e)
            return filtered

        # 2. HIGHEST_RISK
        elif intent == "HIGHEST_RISK":
            max_score = -1
            persons_with_max_score = set()
            for e in retrieved_events:
                score = e.get("risk_score")
                pid = e.get("person_id")
                if pid is not None and pid != "" and score is not None:
                    if int(score) > max_score:
                        max_score = int(score)
            for e in retrieved_events:
                score = e.get("risk_score")
                pid = e.get("person_id")
                if pid is not None and pid != "" and score is not None:
                    if int(score) == max_score:
                        persons_with_max_score.add(normalize_pid(pid))
            
            filtered = []
            for e in retrieved_events:
                pid = e.get("person_id")
                if pid is not None and pid != "":
                    if normalize_pid(pid) in persons_with_max_score:
                        filtered.append(e)
            return filtered

        # 3. ANOMALY_COUNT / ANOMALY_LIST / CAMERA_SUMMARY
        elif intent in ["ANOMALY_COUNT", "ANOMALY_LIST", "CAMERA_SUMMARY"]:
            filtered = []
            for e in retrieved_events:
                etype = e.get("event_type")
                if etype == "anomaly" or etype in ["loitering", "intrusion", "wrong_direction", "sudden_movement", "crowd_anomaly", "abandoned_object"]:
                    filtered.append(e)
            return filtered

        # 4. YES_NO_EVENT
        elif intent == "YES_NO_EVENT":
            filters = self.parse_query_filters(question)
            target_atype = filters.get("anomaly_type")
            target_etype = filters.get("event_type")
            
            filtered = []
            for e in retrieved_events:
                etype = e.get("event_type")
                atype = e.get("anomaly_type")
                if target_atype:
                    if atype == target_atype or etype == target_atype:
                        filtered.append(e)
                elif target_etype:
                    if etype == target_etype:
                        filtered.append(e)
                else:
                    if etype == "anomaly" or etype in ["loitering", "intrusion", "wrong_direction", "sudden_movement", "crowd_anomaly", "abandoned_object"]:
                        filtered.append(e)
            return filtered

        # 5. PERSON_TIMELINE
        elif intent == "PERSON_TIMELINE":
            person_match = re.search(r'\b(?:p|person|pid)\s*[-_]?\s*0*(\d+)\b', query_lower)
            target_pid = person_match.group(1) if person_match else None
            
            if not target_pid:
                filters = self.parse_query_filters(question)
                if filters.get("person_id"):
                    target_pid = filters["person_id"]
            
            filtered = []
            for e in retrieved_events:
                pid = e.get("person_id")
                if pid is not None and pid != "":
                    pid_str = str(pid).strip().lower().lstrip("p").lstrip("0")
                    if target_pid and pid_str == target_pid.lstrip("0"):
                        filtered.append(e)
            return filtered

        # 6. TIME_RANGE_SUMMARY
        elif intent == "TIME_RANGE_SUMMARY":
            filtered = []
            for e in retrieved_events:
                etype = e.get("event_type")
                if etype in ["entry", "exit", "anomaly"] or etype in ["loitering", "intrusion", "wrong_direction", "sudden_movement", "crowd_anomaly", "abandoned_object"]:
                    filtered.append(e)
            return filtered

        # 7. GENERAL / Default
        else:
            filtered = []
            seen_movements = set()
            for e in retrieved_events:
                etype = e.get("event_type")
                if etype in ["entry", "exit", "anomaly"] or etype in ["loitering", "intrusion", "wrong_direction", "sudden_movement", "crowd_anomaly", "abandoned_object"]:
                    filtered.append(e)
                elif etype == "movement":
                    pid = normalize_pid(e.get("person_id"))
                    from_z = e.get("from_zone", "")
                    to_z = e.get("to_zone", "")
                    move_key = f"{pid}_{from_z}_{to_z}"
                    if move_key not in seen_movements:
                        seen_movements.add(move_key)
                        filtered.append(e)
            return filtered
