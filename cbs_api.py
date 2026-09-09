"""
cbs_api.py — CBS Data API + Backend Services
==============================================
Reuses customer_master_data from astor_collection_agent.
Also includes simulated backend services (PTP, Verify, SMS).
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from logger import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "customer_master_data")

cbs_router = APIRouter(prefix="/api/cbs", tags=["CBS Data"])
services_router = APIRouter(prefix="/api/services", tags=["Backend Services"])


# ── CSV Loader ───────────────────────────────────────────────────────────────

def _load_csv(filename: str) -> List[Dict]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


class CBSData:
    def __init__(self):
        self.customers = _load_csv("customer_master_data.csv")
        self.accounts = _load_csv("account_data.csv")
        self.loans = _load_csv("loan_data (1).csv")
        self.transactions = _load_csv("transaction_history.csv")

        self._by_mobile = {c.get("mobile", ""): c for c in self.customers}
        self._by_id = {c.get("customer_id", ""): c for c in self.customers}

        logger.info(f"[CBS] Loaded: {len(self.customers)} customers, {len(self.loans)} loans")

    def get_customer_by_mobile(self, mobile: str) -> Optional[Dict]:
        return self._by_mobile.get(mobile)

    def get_summary(self, mobile: str) -> Optional[Dict]:
        customer = self._by_mobile.get(mobile)
        if not customer:
            return None

        cid = customer.get("customer_id", "")
        loans = [l for l in self.loans if l.get("customer_id") == cid]
        accounts = [a for a in self.accounts if a.get("customer_id") == cid]

        all_txns = []
        for acc in accounts:
            txns = [t for t in self.transactions if t.get("account_no") == acc.get("account_no", "")]
            all_txns.extend(txns[:5])

        total_overdue = sum(float(l.get("emi", 0)) for l in loans if l.get("status", "").lower() == "overdue")

        return {
            "customer": customer,
            "loans": loans,
            "accounts": accounts,
            "transactions": all_txns,
            "total_overdue": total_overdue,
        }


cbs_data = CBSData()


# ── CBS Endpoints ────────────────────────────────────────────────────────────

@cbs_router.get("/customers")
def list_customers():
    return {
        "total": len(cbs_data.customers),
        "customers": [
            {"customer_id": c.get("customer_id"), "name": c.get("name"), "mobile": c.get("mobile"), "city": c.get("city"), "risk_profile": c.get("risk_profile")}
            for c in cbs_data.customers
        ],
    }


@cbs_router.get("/summary/{mobile}")
def get_summary(mobile: str):
    summary = cbs_data.get_summary(mobile)
    if not summary:
        raise HTTPException(status_code=404, detail=f"Customer not found: {mobile}")
    return summary


@cbs_router.get("/transactions/{account_no}")
def get_transactions(account_no: str, limit: int = 10):
    txns = [t for t in cbs_data.transactions if t.get("account_no") == account_no]
    txns.sort(key=lambda x: x.get("date", ""), reverse=True)
    return {"account_no": account_no, "transactions": txns[:limit]}


# ── Backend Services (simulated) ─────────────────────────────────────────────

ptp_store: List[Dict] = []


class VerifyReq(BaseModel):
    customer_id: str
    dob: str


@services_router.post("/verify")
def verify_customer(req: VerifyReq):
    verified = bool(req.dob and len(req.dob) >= 4)
    return {"verified": verified, "customer_id": req.customer_id}


class PTPReq(BaseModel):
    customer_id: str
    ptp_date: str
    ptp_amount: float
    mobile: Optional[str] = None


@services_router.post("/ptp/create")
def create_ptp(req: PTPReq):
    ptp_id = f"ptp_{uuid.uuid4().hex[:6]}"
    record = {
        "ptp_id": ptp_id,
        "customer_id": req.customer_id,
        "ptp_date": req.ptp_date,
        "ptp_amount": req.ptp_amount,
        "status": "Created",
        "created_at": datetime.now().isoformat(),
    }
    ptp_store.append(record)
    logger.info(f"[PTP] Created: {ptp_id} | Amount: {req.ptp_amount}")
    return record


class SMSReq(BaseModel):
    mobile: str
    message: str


@services_router.post("/sms/send")
def send_sms(req: SMSReq):
    sms_id = f"sms_{uuid.uuid4().hex[:6]}"
    logger.info(f"[SMS] Sent to {req.mobile}: {req.message[:50]}...")
    return {"sms_id": sms_id, "mobile": req.mobile, "status": "Sent", "sent_at": datetime.now().isoformat()}
