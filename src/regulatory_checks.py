"""Regulatory checks and startup guidance for FYERS regulatory changes.

This module performs lightweight checks at strategy startup and logs
actionable guidance for the operator when interactive 2FA (daily) or
static IP whitelist is required by the broker.

It does not attempt to perform network calls or refresh tokens
automatically (unless a TOTP secret is present and a non-interactive
TOTP-based flow is explicitly implemented elsewhere).
"""
import logging

log = logging.getLogger(__name__)


def run_regulatory_checks(cfg: dict):
    """Perform startup regulatory checks and log guidance.

    Args:
        cfg: parsed configuration dictionary (contents of config.yaml)

    Returns:
        dict: a small status summary with keys 'ok' (bool) and 'notes' (list).
    """
    notes = []
    ok = True

    fyers = cfg.get("fyers", {}) if cfg else {}
    client_id = fyers.get("client_id")
    secret = fyers.get("secret_key")
    redirect = fyers.get("redirect_uri")
    totp = fyers.get("totp_key")
    static_ips = fyers.get("static_ips") or fyers.get('whitelisted_ips') or fyers.get('static_ip')

    if not client_id or not secret:
        log.error("FYERS client_id or secret_key missing in config/config.yaml.")
        notes.append("Missing client_id or secret_key in config/config.yaml.")
        ok = False
    else:
        log.info("FYERS App ID present: %s", client_id)
        notes.append(f"App ID present: {client_id}")

    if redirect:
        notes.append(f"Redirect URI: {redirect}")

    # Regulatory guidance
    log.warning("FYERS regulatory changes require daily 2FA (interactive) for order placement if no TOTP is available.")
    notes.append("FYERS requires daily 2FA for order placement unless a non-interactive TOTP flow is available.")

    if totp:
        log.info("TOTP key present in config - non-interactive TOTP path may be available if implemented.")
        notes.append("TOTP key present - you may be able to enable non-interactive login if supported and implemented.")
    else:
        log.warning(
            "No TOTP key configured. The operator must perform the interactive 2FA login daily before running live order flows."
        )
        notes.append(
            "No TOTP key: run the interactive auth flow daily (e.g. `python -m src.auth` or use the web redirect) before running live trading.`"
        )

    # Static IP / Whitelist check
    if static_ips:
        log.info("Static/whitelisted IP(s) configured for FYERS: %s", static_ips)
        notes.append(f"Static IPs configured: {static_ips}")
    else:
        log.warning("No static IPs configured for FYERS. FYERS may require App-level IP whitelisting for transactional APIs.")
        notes.append("No static IP configured. Ensure your execution IP(s) are whitelisted in the FYERS dashboard for order placement.")
        # Mark not-ok if missing critical static ip information
        ok = False

    # Static IP whitelist note
    log.warning(
        "Ensure the App ID is activated in FYERS dashboard and your execution IP(s) are whitelisted for order placement."
    )
    notes.append("Ensure App activation and static IP whitelisting in FYERS dashboard for order placement.")

    return {"ok": ok, "notes": notes}
import logging

def check_regulatory_compliance(config):
    """Run a short set of non-invasive checks against strategy config to highlight
    regulatory requirements introduced by FYERS (static IPs, App activation, 2FA/TOTP).
    This only emits operator-visible warnings and does not attempt external validation.
    """
    try:
        fyers_cfg = config.get('fyers', {}) if config else {}
        client_id = fyers_cfg.get('client_id') or fyers_cfg.get('app_id')
        totp = fyers_cfg.get('totp_key')
        static_ips = fyers_cfg.get('static_ips')
        missing = []
        if not client_id:
            missing.append('client_id/app_id')
        if not static_ips:
            logging.warning("[REG-CHK] Static IPs are not configured in config. Order placement requires whitelisted static IPs for the App ID. Please update `config/config.yaml` and activate the App on the FYERS dashboard.")
        else:
            logging.info(f"[REG-CHK] Static IPs configured: {static_ips}")
        if not totp:
            logging.warning("[REG-CHK] No TOTP key configured. FYERS now requires daily 2FA/dedicated auth flow for transactional APIs. Consider configuring a TOTP key in your config or be prepared to run interactive auth daily.")
        else:
            logging.info("[REG-CHK] TOTP key present in config (ensure it's stored securely and access is restricted).")
        if missing:
            logging.warning(f"[REG-CHK] Missing critical config items: {missing}. Ensure your App ID/client_id and redirect URI are present and the App activated on the FYERS dashboard.")
        return True
    except Exception as e:
        logging.exception(f"[REG-CHK] Exception while running regulatory checks: {e}")
        return False
