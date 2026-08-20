# Deployment Checklist (Quick Start)

This checklist highlights the minimal steps and configuration required to run the strategy against FYERS in a compliant and safe manner.

1. App ID / Credentials
   - Ensure a single, active FYERS App ID is configured in `config/config.yaml` under `fyers.app_id`.
   - Confirm the App is activated in FYERS dashboard and the Redirect URI is correct.

2. TOTP / Daily 2FA
   - FYERS requires interactive daily 2FA for order placement unless you provide a non-interactive TOTP key.
   - For live (non-paper) runs, either:
     - Use the interactive web-based auth flow daily (run `python -m src.auth`) before starting the strategy, or
     - Provide a TOTP secret in `config/config.yaml` under `fyers.totp_key` so the process can generate time-based OTPs.
   - If you provide `fyers.totp_key`, keep it secret and rotate as per your ops policy.

3. Static IP / IP Whitelisting
   - FYERS may require App-level IP whitelisting for transactional APIs. Add your execution server's public IP(s) to `config/config.yaml` under `fyers.static_ips`.
   - If you cannot provide a stable IP, run in PAPER mode only.

4. Paper Trading first
   - Always run in `paper_trading: true` mode for the first day. Verify order payloads and sizes in logs under `logs/`.
   - Validate that `execute_trade` and `OrderManager` create bracket orders with `protectionPrice` and `offlineOrder: False` in the payload.

5. Risk & Order sizing
   - Configure `strategy.quantity` and `strategy.lot_size` correctly for the instrument. Defaults are conservative but confirm.
   - Verify risk controls: `strategy.max_trade_duration_minutes`, `strategy.trailing_stop_pct`, and `strategy.min_premium_threshold` match your risk appetite.

6. Network & Secrets
   - Store FYERS credentials (access token or totp_key) in `config/config.yaml` and restrict file permissions.
   - Do not commit secrets to version control.

7. Monitoring & Logging
   - Ensure logs are forwarded or monitored. Key log files: `logs/strategy.log`, `logs/trade_history.csv`, and `logs/trade_history_YYYYMMDD.xlsx`.
   - Watch for WARNINGS about missing TOTP or static IPs on startup.

8. Pre-deployment smoke tests
   - Run the following locally in the virtualenv (paper mode):

```powershell
& ".venv\Scripts\python.exe" -c "import sys; sys.path.append(r'.'); from src.strategy import OpenInterestStrategy; s=OpenInterestStrategy(); s.paper_trading=True; print(s.run_oi_selection_and_place())"
```

9. Rollback plan
   - If unexpected orders are placed, stop the process immediately and cancel open orders via the FYERS dashboard or `src/fyers_api_utils` helper functions.

10. Checklist before switching to LIVE
   - TOTP key present OR interactive auth performed today
   - Static IPs configured and whitelisted in FYERS
   - Paper trades validated for at least one end-to-end day
   - Monitoring alerts configured for exceptions and PnL thresholds

If you want, I can add a small script that validates the `config/config.yaml` for the presence of `fyers.app_id`, `fyers.totp_key` (optional), and `fyers.static_ips` and emits a clear PASS/FAIL report.
