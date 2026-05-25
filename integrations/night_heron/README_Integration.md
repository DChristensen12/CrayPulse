# Night Heron integration

This folder is not part of StrawberryWatch's runtime. Nothing here is imported
or run by the detector, the pipeline, or the tests. It exists to version and
document the code that would be added to the Night Heron production system if
the two are merged.

StrawberryWatch can send its own alerts through src/utils/notifier.py using its
own Gmail style credentials. It does not depend on Night Heron to deliver
anything. The file night_heron_bridge.py holds the code to paste into Night Heron's
email_alerts.py to give that daemon a way to send anomaly alerts in its own
style (Elastic Email SMTP, AlertEvent logging, its multiprocessing worker
pattern). This is for website integration on the software subteam.

Important: this is reference code for the other repository, not a module to run
here. It assumes Night Heron's Django setup, its base.models.AlertEvent, its
EMAIL_HOST_USER / EMAIL_HOST_PASSWORD / FROM_EMAIL credentials, and its existing
_send_sms function. Pasted into StrawberryWatch it would not run, which is why
it lives in its own folder labeled as integration code rather than as a comment
inside notifier.py.

The unbuilt piece is the trigger: how a StrawberryWatch anomaly reaches the
Night Heron daemon. The two clean options are described at the bottom of
night_heron_bridge.py.