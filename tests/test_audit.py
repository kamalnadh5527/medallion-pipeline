from core.audit import AuditLogger


def test_audit_logger_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("core.audit.AUDIT_DIR", tmp_path)
    logger = AuditLogger("audit-test")
    logger.log("test", "started", detail="round trip")

    assert logger.get_logs() == [
        {
            "timestamp": logger.get_logs()[0]["timestamp"],
            "run_id": "audit-test",
            "agent": "test",
            "action": "started",
            "detail": "round trip",
        }
    ]