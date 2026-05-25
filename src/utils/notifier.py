import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()


def send_spill_alert(spill_count, locations_affected):
    sender = os.getenv("ALERT_EMAIL_SENDER")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    receiver = os.getenv("ALERT_EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        print("Alert skipped: Email credentials missing in .env")
        return

    subject = f" ALERT: {spill_count} Potential Spill(s) Detected in Strawberry Creek"
    body = f"""
    The SCMG Anomaly Detection System has identified potential spills.
    Count: {spill_count}
    Affected Locations: {', '.join(locations_affected)}
    Timestamp: {os.popen('date').read()}
    Please check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("Spill alert email sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")


def _diagnosis_line(classification):
    """
    Build the spill type sentence for an anomaly alert from a
    metrics.classify_event result. Handles all three verdicts the classifier
    can return, so the email never claims more certainty than the classifier
    actually has. If classification is None (the classifier was not run for
    this event), returns a neutral line so the alert still reads sensibly.
    """
    if classification is None:
        return "Spill type was not classified for this anomaly."

    verdict = classification.get("verdict")

    if verdict == "diagnosed":
        named = classification.get("named_type", "unknown")
        top = classification["ranked"][0]
        return (
            f"Likely type: {named}. This matched {top['agreements']} of "
            f"{top['comparable']} available water quality parameters. Treat this "
            f"as a lead rather than a confirmation, since confidence depends on "
            f"which sensors were reporting."
        )

    if verdict == "possible_new_type":
        return (
            "The parameter changes did not match any known spill signature. "
            "This may be a new or unclassified event and is worth a closer look."
        )

    # undetermined, or any unexpected verdict
    candidates = classification.get("top_candidates") or []
    hint = ""
    if candidates:
        hint = f" The leading candidate was {candidates[0]}, but this is a hint only."
    return (
        "A spill type could not be determined. The sensors that separate "
        "pollutant types (dissolved oxygen, pH, floating conductivity) were not "
        "reporting at this site." + hint
    )


def send_anomaly_alert(location, score, threshold, event_time, classification=None):
    """
    Send one per-event anomaly alert, stating where and when the anomaly was,
    how far over threshold it scored, and the spill type diagnosis line.

    location, score, threshold, event_time describe a single detected event.
    classification is the dict returned by metrics.classify_event for that
    event, or None if it was not classified. Matches send_spill_alert's
    credential handling and delivery exactly.
    """
    sender = os.getenv("ALERT_EMAIL_SENDER")
    password = os.getenv("ALERT_EMAIL_PASSWORD")
    receiver = os.getenv("ALERT_EMAIL_RECEIVER")

    if not all([sender, password, receiver]):
        print("Alert skipped: Email credentials missing in .env")
        return

    diagnosis = _diagnosis_line(classification)

    if isinstance(event_time, datetime):
        time_str = event_time.isoformat()
    else:
        time_str = str(event_time)

    subject = f" ALERT: Anomaly Detected at {location} in Strawberry Creek"
    body = f"""
    The SCMG Anomaly Detection System has detected an anomaly.
    Location: {location}
    Time: {time_str}
    Anomaly score: {score:.4f} (threshold {threshold:.4f})

    {diagnosis}

    This detection is based on a sudden deviation in conductivity relative to
    the model's prediction of normal creek behavior at this location. Please
    check the latest dashboard visualization for details.
    """

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print(f"Anomaly alert email sent successfully for {location}.")
    except Exception as e:
        print(f"Failed to send anomaly alert email: {e}")


def fire_anomaly_alerts(events):
    """
    End to end helper. Takes a list of detected events and sends one alert per
    event, then also sends the system level summary so the existing
    send_spill_alert behavior is preserved.

    Each event is a dict with keys: location, score, threshold, event_time, and
    optionally classification (a metrics.classify_event result, or None). This
    is what the detection step calls once it has grouped flagged timesteps into
    events and, where possible, classified each one.

    Sending one detailed per-event email plus one summary mirrors how the
    detector thinks: the summary says how many and where, each per-event email
    carries the score and the spill type diagnosis.
    """
    if not events:
        print("No anomaly events to alert on.")
        return

    for ev in events:
        send_anomaly_alert(
            location=ev.get("location", "unknown"),
            score=ev.get("score", float("nan")),
            threshold=ev.get("threshold", float("nan")),
            event_time=ev.get("event_time", datetime.now(timezone.utc)),
            classification=ev.get("classification"),
        )

    locations = sorted({ev.get("location", "unknown") for ev in events})
    send_spill_alert(len(events), locations)
    