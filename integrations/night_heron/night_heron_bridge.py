# Reference code to paste into Night Heron's email_alerts.py.
# NOT runnable inside StrawberryWatch. It assumes the Night Heron environment:
# Django is set up, base.models.AlertEvent exists, the EMAIL_USER / EMAIL_PASS /
# FROM_EMAIL credentials and the _send_sms function from email_alerts.py are in
# scope, and the imports at the top of that file (smtplib, pandas as pd,
# datetime, EmailMessage, typing, logger) are present.
#
# Paste everything below into email_alerts.py, in the alert delivery section.


def _diagnosis_line(classification):
    """
    Spill type sentence for an anomaly alert from a metrics.classify_event
    result. Passed in as a plain dict so this file does not import StrawberryWatch
    code, keeping the two repositories decoupled. Handles all three verdicts so
    the email never overstates the classifier's certainty.
    """
    if classification is None:
        return "Spill type was not classified for this anomaly."

    verdict = classification.get("verdict")

    if verdict == "diagnosed":
        named = classification.get("named_type", "unknown")
        top = classification["ranked"][0]
        return (
            f"Likely type: {named}. Matched {top['agreements']} of "
            f"{top['comparable']} available water quality parameters. Treat as a "
            f"lead, not a confirmation; confidence depends on which sensors "
            f"were reporting."
        )

    if verdict == "possible_new_type":
        return (
            "The parameter changes did not match any known spill signature. "
            "This may be a new or unclassified event and is worth a closer look."
        )

    candidates = classification.get("top_candidates") or []
    hint = ""
    if candidates:
        hint = f" Leading candidate was {candidates[0]}, but this is a hint only."
    return (
        "Spill type could not be determined. The discriminating sensors "
        "(dissolved oxygen, pH, floating conductivity) were not reporting at "
        "this site." + hint
    )


def _send_anomaly_email(rcpts, site, score, threshold, event_time, classification=None):
    """
    Send one GNN anomaly alert email in the same style as _send_email, using the
    same Elastic Email SMTP and EMAIL_USER / EMAIL_PASS / FROM_EMAIL credentials.
    Returns the same status strings _send_email uses.
    """
    if not rcpts:
        logger.info(f"No email recipients for anomaly alert at {site}. Skipping email.")
        return "skipped_no_recipients"
    if not EMAIL_USER or not EMAIL_PASS or not FROM_EMAIL:
        logger.warning("Email credentials not set. Cannot send anomaly email alerts.")
        return "failure_no_creds"

    diagnosis = _diagnosis_line(classification)

    msg = EmailMessage()
    msg["From"], msg["To"] = FROM_EMAIL, ", ".join(rcpts)
    msg["Subject"] = f"Creek Anomaly Alert {site}: score {score:.3f}"
    content = (
        f"An anomaly was detected at {site}.\n\n"
        f"Time: {event_time.isoformat()}\n"
        f"Anomaly score: {score:.4f} (threshold {threshold:.4f})\n\n"
        f"{diagnosis}\n\n"
        f"Detected by the conductivity anomaly model, which flags sudden "
        f"deviations from the predicted normal behavior of the creek at this site."
    )
    msg.set_content(content)

    try:
        with smtplib.SMTP("smtp.elasticemail.com", 2525, timeout=25) as s:
            s.login(EMAIL_USER, EMAIL_PASS)
            s.send_message(msg)
        logger.info(f"Anomaly email alert sent for {site} to {len(rcpts)} recipients.")
        return "success"
    except Exception as e:
        logger.error(f"Failed to send anomaly email alert for {site}: {e}", exc_info=True)
        return "failure_send_error"


def fire_anomaly_alert_task(site, score, threshold, event_time_iso, emails, phones,
                            classification=None, created_by_user_id=None, group_obj_id=None):
    """
    Worker task for a GNN anomaly alert, mirroring fire_alerts_task. Sends the
    email, optionally an SMS, and logs an AlertEvent so anomaly alerts share the
    same audit trail as threshold alerts. event_time is passed as an ISO string
    because, like fire_alerts_task, this runs in a spawned worker and arguments
    must be picklable.
    """
    try:
        django.setup()
        from django.contrib.auth.models import User, Group
        from base.models import AlertEvent
    except RuntimeError as e:
        logger.debug(f"Django already set up in worker process: {e}")
    except Exception as e:
        logger.error(f"Error setting up Django in anomaly alert worker: {e}", exc_info=True)
        return

    event_time = datetime.fromisoformat(event_time_iso)

    detailed_email_status = "pending"
    try:
        if emails:
            detailed_email_status = _send_anomaly_email(emails, site, score, threshold, event_time, classification)
        else:
            detailed_email_status = "skipped_no_recipients"
    except Exception as e_email:
        logger.error(f"Unhandled error sending anomaly email for {site}: {e_email}", exc_info=True)
        detailed_email_status = "failure_exception_calling"
    db_email_status = "success" if detailed_email_status == "success" else "failure"

    # _send_sms expects a numeric Series; the meaningful number for an anomaly
    # is the score, so pass a one element Series of it under the conductivity
    # label since that is what the detector scores on.
    detailed_sms_status = "pending"
    try:
        if phones:
            detailed_sms_status = _send_sms(pd.Series([score]), phones, site, "conductivity")
        else:
            detailed_sms_status = "skipped_no_recipients"
    except Exception as e_sms:
        logger.error(f"Unhandled error sending anomaly SMS for {site}: {e_sms}", exc_info=True)
        detailed_sms_status = "failure_exception_calling"
    db_sms_status = "success" if detailed_sms_status == "success" else "failure"

    named_type = classification.get("named_type") if classification else None
    verdict = classification.get("verdict") if classification else "unclassified"
    notes = f"GNN anomaly alert. Verdict: {verdict}."
    if named_type:
        notes += f" Likely type: {named_type}."

    current_worker_user = None
    current_worker_group = None
    try:
        if created_by_user_id:
            current_worker_user = User.objects.get(id=created_by_user_id)
    except Exception as e:
        logger.error(f"Anomaly worker: error fetching user {created_by_user_id}: {e}", exc_info=True)
    try:
        if group_obj_id:
            current_worker_group = Group.objects.get(id=group_obj_id)
    except Exception as e:
        logger.error(f"Anomaly worker: error fetching group {group_obj_id}: {e}", exc_info=True)

    try:
        AlertEvent.log_event(
            site_code=site,
            sensor_type="conductivity",
            alert_type="gnn_anomaly",
            trigger_value=score,
            rain_pause_applied=False,
            email_status=db_email_status,
            sms_status=db_sms_status,
            notes=notes,
            created_by_user=current_worker_user,
            group_obj=current_worker_group,
        )
    except Exception as e_log:
        logger.error(f"CRITICAL: failed to log GNN anomaly AlertEvent for {site}: {e_log}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# THE TRIGGER: how a StrawberryWatch anomaly reaches this daemon.
#
# Nothing above gets called until something invokes fire_anomaly_alert_task.
# The daemon's main loop only iterates Django rule objects, so it never reaches
# the GNN. Two clean ways to wire it, pick one when the systems actually merge:
#
# Option A, table polling (recommended, fits this daemon's pattern):
#   StrawberryWatch writes each flagged anomaly as a row into a shared MySQL
#   table, e.g. gnn_anomalies, with columns for site, score, threshold,
#   event_time, a JSON classification blob, and a sent flag. Each cycle, this
#   daemon reads unsent rows and submits them to the pool, the same way it fires
#   rule alerts:
#
#     for row in unsent_gnn_anomaly_rows():
#         ALERT_POOL.apply_async(
#             fire_anomaly_alert_task,
#             args=(row["site"], row["score"], row["threshold"],
#                   row["event_time"].isoformat(),
#                   recipients_for(row["site"]), phones_for(row["site"]),
#                   json.loads(row["classification"]) if row["classification"] else None,
#                   system_user.id if system_user else None,
#                   alerts_group.id if alerts_group else None),
#         )
#         mark_row_sent(row["id"])
#
#   This keeps the daemon the single owner of delivery and the AlertEvent log,
#   and StrawberryWatch never needs Night Heron's credentials.
#
# Option B, direct call:
#   StrawberryWatch imports and calls _send_anomaly_email itself with Night
#   Heron's credentials. Simpler but couples the repos and splits alert
#   ownership across two systems, so Option A is likely preferred.
# ─────────────────────────────────────────────────────────────────────────────
