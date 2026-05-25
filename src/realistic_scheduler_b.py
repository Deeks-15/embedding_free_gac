"""Held-out corpus (b) — same regime as realistic_scheduler.py, deliberately
varied surface for the generalization test.

Changes vs corpus (a):
  1. SERVICE NAME SWAPS — every dominant service is renamed:
       auth-svc       → identity-svc
       payment-svc    → billing-svc
       inventory-svc  → stock-svc
       checkout-svc   → cart-svc
       recommender-svc→ suggest-svc
       notification-svc → alerting-svc
       worker-email   → jobrunner-mail
       worker-imageproc → jobrunner-media
       OrderController → PurchaseController (Java)
  2. ZIPFIAN WEIGHT SHIFT — payment/billing dominate now; nginx static deweighted;
     redis weighted lower; some long-tail rare templates boosted slightly.
  3. NEW LONG-TAIL TEMPLATES (5) that did NOT exist in (a):
       inventory adjustment (manual stock correction)
       returns workflow (refund initiated by customer)
       A/B test exposure (user assigned to experiment arm)
       customer-support ticket events
       batch job retry-after-failure
  4. REWORDED MESSAGES — many keywords paraphrased:
       "order placed successfully" → "purchase committed"
       "payment declined" → "transaction rejected"
       "user authenticated" → "identity verified"
       "stock running low" → "merchandise below reorder line"
       "container exceeded memory" → "container OOM-killed"
  5. (a) had 46 templates; (b) has 49 (+ rewording shifts vocabulary).

FROZEN: this generator is built once. The matching driver
(phase2_heldout.py) runs C0 vs C3 once. No re-tuning.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import signal
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "logs/realistic_b.log"
PID_FILE = Path(__file__).resolve().parent.parent / "logs/realistic_b.pid"


# ---------- shared helpers (same as corpus a) ----------------------------

def rid(n=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def ip():
    return f"{random.choice([10, 172, 192, 203])}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def public_ip():
    return f"{random.randint(2, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def amount():
    return round(random.uniform(5.0, 4999.99), 2)

def user_id(): return f"u_{rid(8)}"
def order_id(): return f"prc_{rid(10)}"   # changed prefix from ord_→prc_ for held-out
def session_id(): return rid(24)
def request_id(): return f"req_{rid(12)}"

def ts_iso(precision="ms"):
    now = datetime.now(timezone.utc)
    if precision == "ms":
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")

def ts_apache():
    return datetime.now(timezone.utc).strftime("%d/%b/%Y:%H:%M:%S +0000")

def ts_postgres():
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d} UTC"

def ts_java():
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond // 1000:03d}"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_3) AppleWebKit/605.1.15 Version/17.2.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "ShopAppMobile/4.12.1 (iOS 17.3; iPhone14,2)",
    "ShopAppMobile/4.12.1 (Android 14; Pixel 8)",
    "curl/8.4.0",
    "PostmanRuntime/7.36.0",
    "GoogleBot/2.1 (+http://www.google.com/bot.html)",
]


# ---------- TEMPLATES (renamed services, reworded messages) -------------

# Apache (nginx) — slight rewording, same shape
def t_nginx_get_product(level):
    code = random.choices([200, 200, 200, 304, 404, 500], weights=[10, 10, 10, 3, 2, 1])[0]
    return (f'{public_ip()} - - [{ts_apache()}] '
            f'"GET /catalog/items/{random.randint(1000, 99999)} HTTP/1.1" {code} '
            f'{random.randint(200, 8000)} "https://store.example.com/" "{random.choice(USER_AGENTS)}"')

def t_nginx_post_checkout(level):
    code = random.choices([200, 201, 400, 402, 500], weights=[15, 10, 4, 2, 2])[0]
    return (f'{public_ip()} - - [{ts_apache()}] '
            f'"POST /cart/confirm HTTP/1.1" {code} '
            f'{random.randint(50, 2000)} "https://store.example.com/cart" "{random.choice(USER_AGENTS)}"')

def t_nginx_search(level):
    queries = ["red trainers", "tablet case", "laptop sleeve", "coffee table",
               "wireless mouse", "kitchen knife set", "garden hose",
               "smart watch", "running shorts"]
    code = random.choice([200, 200, 200, 200, 304, 500])
    q = random.choice(queries).replace(" ", "%20")
    return (f'{public_ip()} - - [{ts_apache()}] '
            f'"GET /catalog/lookup?term={q}&page={random.randint(1, 12)} HTTP/1.1" {code} '
            f'{random.randint(500, 12000)} "https://store.example.com/search" "{random.choice(USER_AGENTS)}"')

def t_nginx_static(level):
    paths = ["/static/css/main.7a3b.css", "/static/js/bundle.4c2d.js",
             "/static/img/brand.png", "/static/fonts/inter.woff2",
             "/favicon.ico"]
    code = random.choice([200, 200, 304, 304, 304])
    return (f'{public_ip()} - - [{ts_apache()}] '
            f'"GET {random.choice(paths)} HTTP/1.1" {code} '
            f'{random.randint(1000, 200000)} "-" "{random.choice(USER_AGENTS)}"')


# Java — PurchaseController (was OrderController in a)
def t_java_request_ok(level):
    return (f"{ts_java()} [http-nio-8080-exec-{random.randint(1, 64)}] INFO  "
            f"c.example.api.PurchaseController - Request handled: "
            f"purchase={order_id()} user={user_id()} duration={random.randint(5, 250)}ms")

def t_java_validation_warn(level):
    return (f"{ts_java()} [http-nio-8080-exec-{random.randint(1, 64)}] WARN  "
            f"c.example.api.PurchaseController - Input rejected: "
            f"field={random.choice(['email', 'shipping_address', 'card_number', 'cvv'])} "
            f"reason={random.choice(['invalid_format', 'required', 'too_long', 'malformed'])}")

def t_java_npe(level):
    thread = f"http-nio-8080-exec-{random.randint(1, 64)}"
    head = (f"{ts_java()} [{thread}] ERROR c.example.api.PurchaseController - "
            f"Unexpected error processing purchase {order_id()}")
    trace = [
        "java.lang.NullPointerException: Cannot invoke \"com.example.model.Basket.getItems()\" because \"basket\" is null",
        "\tat com.example.api.PurchaseController.commitPurchase(PurchaseController.java:142)",
        "\tat com.example.api.PurchaseController$$FastClassBySpringCGLIB$$abc123.invoke(<generated>)",
        "\tat org.springframework.cglib.proxy.MethodProxy.invoke(MethodProxy.java:218)",
        "\tat org.springframework.aop.framework.CglibAopProxy$CglibMethodInvocation.invokeJoinpoint(CglibAopProxy.java:783)",
        "\tat org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:163)",
        "\tat com.example.tracing.TraceAspect.aroundExecution(TraceAspect.java:62)",
        "\tat sun.reflect.GeneratedMethodAccessor134.invoke(Unknown Source)",
        "\tat java.lang.reflect.Method.invoke(Method.java:498)",
        "\tat org.springframework.aop.aspectj.AspectJAroundAdvice.invoke(AspectJAroundAdvice.java:72)",
        "\tat org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:175)",
        "\tat com.example.api.PurchaseController$$EnhancerBySpringCGLIB$$xyz789.commitPurchase(<generated>)",
        "\tat org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:205)",
        "\tat org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod.invokeAndHandle(ServletInvocableHandlerMethod.java:117)",
        "\tat org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1057)",
    ]
    return [head] + trace

def t_java_db_timeout(level):
    head = (f"{ts_java()} [http-nio-8080-exec-{random.randint(1, 64)}] WARN  "
            f"c.example.persistence.PurchaseRepository - Query SLA breach: "
            f"queryId=q_{rid(8)} duration={random.randint(2000, 8000)}ms")
    trace = [
        "org.springframework.dao.QueryTimeoutException: PreparedStatementCallback; "
        "SQL [SELECT p.* FROM purchases p WHERE p.user_id = ? AND p.created_at > ?]",
        "\tat org.springframework.jdbc.support.SQLErrorCodeSQLExceptionTranslator.doTranslate(SQLErrorCodeSQLExceptionTranslator.java:259)",
        "\tat org.springframework.jdbc.core.JdbcTemplate.execute(JdbcTemplate.java:649)",
        "Caused by: org.postgresql.util.PSQLException: ERROR: canceling statement due to statement timeout",
        "\tat org.postgresql.core.v3.QueryExecutorImpl.receiveErrorResponse(QueryExecutorImpl.java:2675)",
    ]
    return [head] + trace


# JSON — renamed services + reworded msgs
def t_json_checkout_ok(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "cart-svc",
        "msg": "purchase committed",
        "purchase_id": order_id(), "user_id": user_id(),
        "amount": amount(), "currency": random.choice(["USD", "EUR", "GBP"]),
        "payment_method": random.choice(["card", "paypal", "applepay", "googlepay"]),
        "items": random.randint(1, 8), "trace_id": rid(16),
    }
    return json.dumps(obj)

def t_json_payment_declined(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "billing-svc",
        "msg": "transaction rejected", "purchase_id": order_id(),
        "user_id": user_id(), "amount": amount(),
        "reason": random.choice(["insufficient_funds", "card_expired",
                                  "fraud_suspicion", "issuer_decline", "3ds_failed"]),
        "gateway": random.choice(["stripe", "adyen", "braintree"]),
        "trace_id": rid(16),
    }
    return json.dumps(obj)

def t_json_inventory_low(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "stock-svc",
        "msg": "merchandise below reorder line",
        "sku": f"SKU-{random.randint(10000, 99999)}",
        "warehouse": random.choice(["us-east", "us-west", "eu-central", "ap-south"]),
        "current_qty": random.randint(0, 10),
        "reorder_threshold": random.randint(10, 50),
    }
    return json.dumps(obj)

def t_json_inventory_ok(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "stock-svc",
        "msg": "merchandise reserved",
        "sku": f"SKU-{random.randint(10000, 99999)}",
        "qty": random.randint(1, 8),
        "purchase_id": order_id(),
    }
    return json.dumps(obj)

def t_json_auth_login(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "identity-svc",
        "msg": "identity verified",
        "user_id": user_id(),
        "method": random.choice(["password", "oauth_google", "oauth_apple", "sms_otp"]),
        "ip": public_ip(), "session_id": session_id(),
    }
    return json.dumps(obj)

def t_json_auth_failed(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "identity-svc",
        "msg": "identity verification denied",
        "user_id": user_id(), "method": "password",
        "reason": random.choice(["invalid_credentials", "account_locked",
                                  "mfa_required", "ip_blocked"]),
        "ip": public_ip(), "attempts": random.randint(1, 8),
    }
    return json.dumps(obj)


# k8s — reworded events
def t_kv_k8s_pod_scheduled(level):
    return (f"time={ts_iso('s')} level=info source=scheduler "
            f"event=Scheduled namespace=production "
            f"pod={random.choice(['cart', 'stock', 'identity', 'lookup', 'basket'])}"
            f"-{rid(5)} "
            f"node=ip-10-0-{random.randint(1, 99)}-{random.randint(1, 254)} "
            f"msg=\"Successfully bound pod to node\"")

def t_kv_k8s_pod_oom(level):
    return (f"time={ts_iso('s')} level=warn source=kubelet "
            f"event=OOMKilled namespace=production "
            f"pod={random.choice(['cart', 'stock', 'identity', 'lookup'])}"
            f"-{rid(5)} "
            f"container={random.choice(['app', 'sidecar'])} "
            f"reason=container_oom_killed "
            f"memory_usage_mb={random.randint(512, 2048)} "
            f"limit_mb={random.choice([512, 1024, 2048])} "
            f"msg=\"Container OOM-killed: memory limit exceeded\"")

def t_kv_k8s_image_pull(level):
    return (f"time={ts_iso('s')} level=info source=kubelet "
            f"event=Pulled namespace=production "
            f"pod={random.choice(['cart', 'stock', 'identity'])}-{rid(5)} "
            f"image=ecr/{random.choice(['cart', 'stock', 'identity'])}-svc:v{random.randint(100, 999)}.{random.randint(0, 99)} "
            f"msg=\"Successfully pulled image\"")


# Postgres
def t_postgres_slow_query(level):
    duration = random.randint(1500, 6500)
    head = (f"{ts_postgres()} [{random.randint(1000, 9999)}] LOG:  "
            f"duration: {duration}.{random.randint(100, 999)} ms  "
            f"statement: SELECT p.id, p.user_id, p.total, COUNT(pi.id) AS item_count "
            f"FROM purchases p LEFT JOIN purchase_items pi ON pi.purchase_id = p.id "
            f"WHERE p.created_at > '{datetime.now(timezone.utc).strftime('%Y-%m-%d')}' "
            f"AND p.status = 'pending' GROUP BY p.id ORDER BY p.created_at DESC LIMIT 100")
    plan = [
        f"{ts_postgres()} [{random.randint(1000, 9999)}] DETAIL:  "
        f"Limit  (cost=158234.45..158234.70 rows=100 width=64) (actual time={duration-50}..{duration} rows=100 loops=1)",
        f"          ->  Sort  (cost=158234.45..158734.45 rows=200000 width=64) (actual time={duration-100}..{duration-10} rows=100 loops=1)",
        f"                Sort Key: p.created_at DESC",
        f"                Sort Method: top-N heapsort  Memory: 32kB",
        f"                ->  GroupAggregate  (cost=125000.00..148000.00 rows=200000 width=64) (actual time={duration-500}..{duration-50} rows=180432 loops=1)",
        f"                      ->  Hash Left Join  (cost=15000.00..120000.00 rows=2500000 width=48) (actual time=100..{duration-200} rows=2412341 loops=1)",
        f"                            Hash Cond: (pi.purchase_id = p.id)",
        f"                            ->  Seq Scan on purchase_items pi  (cost=0.00..50000.00 rows=2500000 width=16)",
        f"                            ->  Hash  (cost=12000.00..12000.00 rows=200000 width=40)",
    ]
    return [head] + plan

def t_postgres_deadlock(level):
    head = f"{ts_postgres()} [{random.randint(1000, 9999)}] ERROR:  deadlock detected"
    detail = [
        f"{ts_postgres()} [{random.randint(1000, 9999)}] DETAIL:  "
        f"Process 4521 waits for ShareLock on transaction 9821345; blocked by process 4823.",
        f"Process 4823 waits for ShareLock on transaction 9821456; blocked by process 4521.",
        f"Process 4521: UPDATE purchases SET status = 'paid' WHERE id = '{order_id()}'",
        f"Process 4823: UPDATE purchases SET status = 'cancelled' WHERE id = '{order_id()}'",
        f"{ts_postgres()} [{random.randint(1000, 9999)}] HINT:  See server log for query details.",
    ]
    return [head] + detail

def t_postgres_query_ok(level):
    return (f"{ts_postgres()} [{random.randint(1000, 9999)}] LOG:  "
            f"duration: {random.randint(2, 80)}.{random.randint(100, 999)} ms  "
            f"statement: SELECT * FROM "
            f"{random.choice(['shoppers', 'merchandise', 'purchases', 'basket_items', 'ratings'])} "
            f"WHERE id = '{rid(8)}'")


# Redis
def t_redis_command(level):
    return (f"[{random.randint(1, 32)}] {ts_iso('s')} "
            f"{random.choice(['*', '#', '-'])} "
            f"{random.choice(['GET', 'SET', 'DEL', 'HGET', 'LPUSH', 'ZADD', 'EXPIRE'])} "
            f"{random.choice(['session', 'basket', 'item', 'shopper', 'stock'])}:"
            f"{rid(10)} ({random.randint(0, 5)} us)")


# Elasticsearch — paths/services tweaked
def t_elasticsearch_query(level):
    return (f"[{ts_iso('s')}] [INFO] [o.e.r.RestController] [node-{random.randint(1, 6)}] "
            f"received [GET] request for [/merchandise/_search] from "
            f"[{ip()}], took [{random.randint(5, 350)}ms]")

def t_elasticsearch_unassigned(level):
    return (f"[{ts_iso('s')}] [WARN] [o.e.c.r.a.AllocationService] [master-1] "
            f"shard [{random.choice(['merchandise', 'purchases', 'shoppers', 'ratings'])}]"
            f"[{random.randint(0, 7)}] cannot be allocated: "
            f"reason={random.choice(['ALLOCATION_FAILED', 'DELAYED_ALLOCATION', 'NODE_LEFT'])}, "
            f"node={random.choice(['data-1', 'data-2', 'data-3'])}")


# Mobile crash — reworded a bit
def t_mobile_crash(level):
    head = (f"{ts_iso()} [mobile-telemetry] CRASH "
            f"app=ShopApp/4.12.1 platform={random.choice(['iOS 17.3', 'Android 14'])} "
            f"device={random.choice(['iPhone14,2', 'iPhone15,3', 'Pixel 8', 'SM-S928U'])} "
            f"user_id={user_id()}")
    detail = [
        f"  Exception: {random.choice(['NSInvalidArgumentException', 'NullPointerException', 'OutOfMemoryError'])}",
        f"  Reason: {random.choice(['attempt to insert nil object', 'index out of range', 'unable to decode response'])}",
        f"  Stack:",
        f"    0  ShopApp                      0x000000010234abcd -[CartVC commitPurchase:] + 412",
        f"    1  ShopApp                      0x000000010234ef12 -[CartVC viewDidAppear:] + 89",
        f"    2  UIKitCore                    0x00007fff48a12345 -[UIViewController _setViewAppearState:] + 712",
        f"    3  UIKitCore                    0x00007fff48a23456 -[UIView _addSubview:positioned:relativeTo:] + 1245",
        f"    4  libdispatch.dylib            0x00007fff513def01 _dispatch_call_block_and_release + 12",
        f"  Battery: {random.randint(5, 100)}%  Memory: {random.randint(100, 800)}MB / {random.randint(2048, 8192)}MB",
        f"  Network: {random.choice(['WiFi', 'LTE', '5G'])}  Locale: {random.choice(['en_US', 'es_ES', 'ja_JP', 'de_DE'])}",
    ]
    return [head] + detail


# Audit (renamed event names)
def t_audit_login(level):
    obj = {"ts": ts_iso(), "level": "INFO", "category": "audit",
           "event": "shopper.session_started", "actor": user_id(),
           "ip": public_ip(), "country": random.choice(["US", "UK", "DE", "JP", "IN", "BR"]),
           "result": "success"}
    return json.dumps(obj)

def t_audit_privilege_escalation(level):
    obj = {"ts": ts_iso(), "level": "WARN", "category": "audit",
           "event": "role.elevation_attempt", "actor": user_id(),
           "target_role": random.choice(["admin", "merchant_admin", "support_agent"]),
           "ip": public_ip(),
           "result": "denied", "reason": "insufficient_permissions"}
    return json.dumps(obj)

def t_audit_data_export(level):
    obj = {"ts": ts_iso(), "level": "INFO", "category": "audit",
           "event": "data.bulk_export", "actor": user_id(),
           "dataset": random.choice(["purchases", "shoppers", "transactions"]),
           "rows_exported": random.randint(100, 50000),
           "destination": random.choice(["s3://reports/", "user_download", "scheduled_email"])}
    return json.dumps(obj)


# CDN
def t_cdn_cache_miss(level):
    return (f'"{ts_iso()}",cf-pop="{random.choice(["DFW", "LHR", "NRT", "SIN", "FRA"])}",'
            f'cache_status="MISS",origin_status={random.randint(200, 599)},'
            f'origin_time_ms={random.randint(50, 2000)},'
            f'edge_time_ms={random.randint(1, 50)},'
            f'host="cdn.store.example.com",'
            f'uri="/static/items/{random.randint(1000, 99999)}.jpg",'
            f'bytes={random.randint(5000, 500000)}')

def t_cdn_ddos_block(level):
    return (f'"{ts_iso()}",cf-pop="{random.choice(["DFW", "LHR", "NRT"])}",'
            f'cache_status="BLOCK",origin_status=0,'
            f'rule_id="ddos_{rid(6)}",'
            f'client_ip="{public_ip()}",'
            f'reason="rate_limit_exceeded",'
            f'request_count={random.randint(1000, 50000)}')


# Workers — jobrunner-* instead of worker-*
def t_worker_email_sent(level):
    return (f"{ts_iso()} [jobrunner-mail] INFO Email dispatched: "
            f"template={random.choice(['purchase_confirmation', 'shipping_update', 'password_reset', 'basket_abandoned', 'welcome'])} "
            f"to={user_id()} provider={random.choice(['sendgrid', 'ses', 'mailgun'])}")

def t_worker_image_resize(level):
    return (f"{ts_iso()} [jobrunner-media] INFO Image transcoded: "
            f"sku=SKU-{random.randint(10000, 99999)} "
            f"variants={random.randint(3, 8)} "
            f"duration_ms={random.randint(50, 2500)} "
            f"output_bytes={random.randint(10000, 800000)}")


# Recommender — suggest-svc
def t_recommender_inference(level):
    obj = {"ts": ts_iso(), "level": "INFO", "service": "suggest-svc",
           "msg": "suggestions computed",
           "user_id": user_id(), "context": random.choice(["homepage", "product_page", "cart", "email"]),
           "model_version": f"v{random.randint(20, 99)}.{random.randint(0, 9)}",
           "latency_ms": random.randint(20, 200),
           "n_results": random.randint(5, 50)}
    return json.dumps(obj)

def t_recommender_cold_start(level):
    obj = {"ts": ts_iso(), "level": "WARN", "service": "suggest-svc",
           "msg": "cold-start fallback engaged",
           "user_id": user_id(), "reason": "no_interaction_history",
           "fallback": random.choice(["bestsellers", "trending", "category_default"])}
    return json.dumps(obj)


# Notification — alerting-svc
def t_notif_sms_delay(level):
    return (f"{ts_iso()} [alerting-svc] WARN SMS dispatch backlog: "
            f"queue_depth={random.randint(50, 500)} avg_age_s={random.randint(30, 600)}")

def t_notif_smtp_fail(level):
    return (f"{ts_iso()} [alerting-svc] ERROR Mail server unreachable: "
            f"host={random.choice(['smtp.sendgrid.com', 'smtp.ses.com', 'smtp.mailgun.com'])} "
            f"err={random.choice(['conn_refused', 'tls_handshake_failed', 'timeout'])}")


# ===== NEW long-tail templates that don't exist in (a) ==================

def t_new_inventory_adjustment(level):
    obj = {"ts": ts_iso(), "level": "INFO", "service": "stock-svc",
           "msg": "manual stock adjustment recorded",
           "sku": f"SKU-{random.randint(10000, 99999)}",
           "delta": random.randint(-20, 50),
           "reason": random.choice(["physical_count", "damage_writeoff", "transfer_in", "supplier_correction"]),
           "operator": f"emp_{rid(6)}"}
    return json.dumps(obj)

def t_new_returns_initiated(level):
    obj = {"ts": ts_iso(), "level": "INFO", "service": "billing-svc",
           "msg": "return workflow initiated by customer",
           "purchase_id": order_id(), "return_id": f"rtn_{rid(8)}",
           "reason_code": random.choice(["damaged_item", "wrong_size", "not_as_described", "changed_mind"]),
           "refund_amount": amount()}
    return json.dumps(obj)

def t_new_experiment_exposure(level):
    obj = {"ts": ts_iso(), "level": "INFO", "service": "ab-testing",
           "msg": "user assigned to experiment arm",
           "experiment_id": f"exp_{rid(6)}",
           "arm": random.choice(["control", "variant_a", "variant_b"]),
           "user_id": user_id(),
           "feature_flag": random.choice(["new_checkout_flow", "ai_recommendations", "express_delivery", "instant_payment"])}
    return json.dumps(obj)

def t_new_support_ticket(level):
    obj = {"ts": ts_iso(), "level": "INFO", "service": "helpdesk-svc",
           "msg": "support ticket opened",
           "ticket_id": f"tkt_{rid(8)}",
           "user_id": user_id(),
           "category": random.choice(["delivery_question", "refund_request", "product_defect", "account_access"]),
           "priority": random.choice(["low", "normal", "high", "urgent"])}
    return json.dumps(obj)

def t_new_batch_retry(level):
    return (f"{ts_iso()} [jobrunner-batch] WARN Job retry scheduled: "
            f"job_type={random.choice(['nightly-export', 'price-sync', 'inventory-reconcile', 'invoice-batch'])} "
            f"attempt={random.randint(2, 5)} backoff_s={random.randint(30, 600)} "
            f"prev_error={random.choice(['ConnectionTimeout', 'RateLimited', 'DownstreamUnavailable'])}")


# ---------- registry with SHIFTED Zipfian weights -----------------------

# Compared to corpus (a):
#   - payment/billing dominate (used to be lower weight); reflects an
#     e-commerce platform where finance events dominate
#   - nginx static deweighted (less noise)
#   - redis lower (caches are quieter here)
#   - new long-tail templates given mid-weight (3-6)
TEMPLATES = [
    # Top tier - payment+identity dominant now
    ("billing_purchase_ok",      "INFO",  120, t_json_checkout_ok),       # was 45 in (a)
    ("identity_login_ok",        "INFO",  100, t_json_auth_login),         # was 55
    ("stock_reserved",           "INFO",   80, t_json_inventory_ok),       # was 80
    ("nginx_get_product",        "INFO",   70, t_nginx_get_product),       # was 120
    ("postgres_query_ok",        "INFO",   65, t_postgres_query_ok),
    ("k8s_image_pull",           "INFO",   55, t_kv_k8s_image_pull),       # was 70
    ("redis_command",            "INFO",   45, t_redis_command),           # was 90
    ("nginx_search",             "INFO",   45, t_nginx_search),
    ("java_request_ok",          "INFO",   45, t_java_request_ok),
    ("nginx_static",             "INFO",   35, t_nginx_static),            # was 100, deweighted
    ("es_query",                 "INFO",   35, t_elasticsearch_query),
    ("suggest_ok",               "INFO",   30, t_recommender_inference),
    ("audit_login",              "INFO",   25, t_audit_login),
    ("cdn_cache_miss",           "INFO",   25, t_cdn_cache_miss),
    ("mail_dispatched",          "INFO",   22, t_worker_email_sent),
    ("media_resized",            "INFO",   22, t_worker_image_resize),

    # Mid tier - operational warnings
    ("nginx_checkout",           "INFO",   20, t_nginx_post_checkout),
    ("k8s_scheduled",            "INFO",   15, t_kv_k8s_pod_scheduled),
    ("stock_low",                "WARN",   15, t_json_inventory_low),
    ("billing_declined",         "WARN",   15, t_json_payment_declined),   # boosted from 10
    ("java_validation",          "WARN",   12, t_java_validation_warn),
    ("identity_failed",          "WARN",   10, t_json_auth_failed),
    ("es_unassigned",            "WARN",    8, t_elasticsearch_unassigned),
    ("suggest_cold_start",       "WARN",    6, t_recommender_cold_start),
    ("audit_data_export",        "INFO",    5, t_audit_data_export),
    ("postgres_slow",            "WARN",    5, t_postgres_slow_query),

    # Lower tier - errors + new long-tail
    ("new_experiment",           "INFO",    6, t_new_experiment_exposure),  # NEW (a-)
    ("new_support_ticket",       "INFO",    6, t_new_support_ticket),       # NEW
    ("new_returns_init",         "INFO",    5, t_new_returns_initiated),    # NEW
    ("new_inventory_adj",        "INFO",    4, t_new_inventory_adjustment),  # NEW
    ("new_batch_retry",          "WARN",    3, t_new_batch_retry),           # NEW
    ("java_db_timeout",          "WARN",    4, t_java_db_timeout),
    ("java_npe",                 "ERROR",   3, t_java_npe),
    ("mobile_crash",             "ERROR",   3, t_mobile_crash),
    ("k8s_oom",                  "WARN",    3, t_kv_k8s_pod_oom),
    ("postgres_deadlock",        "ERROR",   2, t_postgres_deadlock),
    ("cdn_ddos",                 "WARN",    2, t_cdn_ddos_block),
    ("audit_privilege",          "WARN",    2, t_audit_privilege_escalation),
    ("sms_delay",                "WARN",    2, t_notif_sms_delay),
    ("smtp_fail",                "ERROR",   2, t_notif_smtp_fail),
]


def template_weights():
    return [w for _, _, w, _ in TEMPLATES]


def malformed_line():
    kind = random.choice(["truncated", "empty", "garbage", "half_json", "wrong_encoding"])
    if kind == "truncated":
        _, _, _, render = random.choices(TEMPLATES, weights=template_weights())[0]
        result = render("INFO")
        line = result if isinstance(result, str) else result[0]
        cut = random.randint(20, max(20, len(line) // 2))
        return line[:cut]
    elif kind == "empty":
        return ""
    elif kind == "garbage":
        return f"{ts_iso()} {random.choice(['<<<', '###', '???'])} " + "".join(
            random.choices(string.printable, k=random.randint(20, 80))
        ).replace("\n", "")
    elif kind == "half_json":
        partial = {"ts": ts_iso(), "level": "ERROR", "service": "unknown",
                   "msg": "partial entry — caller did not flush"}
        s = json.dumps(partial)
        return s[:random.randint(20, len(s) - 5)]
    else:
        return f"{ts_iso()} [?service?] some message with bad bytes: \\xfd\\xff\\xfe (encoding fallback)"


def emit(out):
    if random.random() < 0.05:
        out.write(malformed_line() + "\n")
        out.flush()
        return 1
    _, _, _, render = random.choices(TEMPLATES, weights=template_weights())[0]
    result = render("INFO")
    if isinstance(result, list):
        for line in result:
            out.write(line + "\n")
        out.flush()
        return len(result)
    else:
        out.write(result + "\n")
        out.flush()
        return 1


def run(rate, duration):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    interval = 1.0 / rate
    start = time.time()
    written_lines = 0
    written_entries = 0

    def stop_handler(sig, frame):
        try: PID_FILE.unlink()
        except FileNotFoundError: pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    print(f"[scheduler-b] writing held-out log to {LOG_FILE} at ~{rate} lines/sec "
          f"(duration: {duration}s)")
    print(f"[scheduler-b] {len(TEMPLATES)} templates "
          f"({sum(1 for _, _, w, _ in TEMPLATES if w >= 20)} dominant, "
          f"{sum(1 for _, _, w, _ in TEMPLATES if w <= 6)} long-tail)")
    print(f"[scheduler-b] FROZEN corpus — do not peek at failures")
    with open(LOG_FILE, "a", buffering=1) as out:
        next_t = time.time()
        while True:
            if duration is not None and (time.time() - start) >= duration:
                break
            n = emit(out)
            written_lines += n
            written_entries += 1
            if written_entries % 1000 == 0:
                elapsed = time.time() - start
                print(f"[scheduler-b] +{written_entries} entries ({written_lines} raw lines, "
                      f"{written_lines/elapsed:.0f} lines/s)")
            next_t += interval * n
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
    print(f"[scheduler-b] done. {written_entries} entries / {written_lines} raw lines in "
          f"{time.time()-start:.1f}s")
    try: PID_FILE.unlink()
    except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=200)
    ap.add_argument("--duration", type=float, default=150)
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()
    if args.truncate and LOG_FILE.exists():
        LOG_FILE.write_text("")
    run(args.rate, args.duration)


if __name__ == "__main__":
    main()
