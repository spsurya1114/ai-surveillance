
import json


def event_to_text(event):
    etype = event["event_type"]
    pid = event["person_id"]
    cam = event["camera_id"]
    time = event["timestamp"]

    if etype == "entry":
        return f"At {time} seconds, person {pid} entered {event['to_zone']} zone in {cam}."

    elif etype == "movement":
        return f"At {time} seconds, person {pid} moved from {event['from_zone']} to {event['to_zone']} in {cam}."

    elif etype == "exit":
        return f"At {time} seconds, person {pid} exited the scene from {event['from_zone']} in {cam}."

    return ""


def convert_logs_to_text(log_path):
    with open(log_path, "r") as f:
        events = json.load(f)

    texts = []

    for event in events:
        sentence = event_to_text(event)
        if sentence:
            texts.append(sentence)

    return texts
