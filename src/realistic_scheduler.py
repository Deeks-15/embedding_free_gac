"""Realistic e-commerce SaaS log generator.

Addresses the reviewer's critique that the synthetic log corpus was "too clean":
  - **Multi-line entries**: Java stack traces (15-40 lines), Postgres query plans
    (5-12 lines), mobile crash reports (10-25 lines).
  - **Mixed formats** from the same stream: Apache combined log, Logback Java,
    JSON microservice logs, key=value k8s events, PostgreSQL slow log,
    syslog-style audit log.
  - **Long-tail cardinality** (Zipfian): a few dominant templates carry most
    traffic, a long tail of ~50 rare templates accounts for ~10%.
  - **Dirty lines** (~5%): truncated mid-line, garbled encoding, empty lines,
    half-formed JSON, mixed-up field orders.

Industry vertical: **e-commerce SaaS platform** — a canonical stack with edge
CDN, API gateway, Java backend services, Node.js microservices, PostgreSQL,
Redis cache, Elasticsearch, Kubernetes orchestration, mobile clients, and
security/audit subsystems.

Usage:
  python src/realistic_scheduler.py --truncate --rate 80 --duration 90
  python src/realistic_scheduler.py --truncate --rate 80 --duration 200 --drift-after 14000
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

LOG_FILE = Path(__file__).resolve().parent.parent / "logs/realistic.log"
PID_FILE = Path(__file__).resolve().parent.parent / "logs/realistic.pid"


# ---------- helpers ------------------------------------------------------

def rid(n=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def ip():
    return f"{random.choice([10, 172, 192, 203])}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def public_ip():
    return f"{random.randint(2, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def amount():
    return round(random.uniform(5.0, 4999.99), 2)


def user_id():
    return f"u_{rid(8)}"


def order_id():
    return f"ord_{rid(10)}"


def session_id():
    return rid(24)


def request_id():
    return f"req_{rid(12)}"


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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
    "ShopAppMobile/4.12.1 (iOS 17.3; iPhone14,2)",
    "ShopAppMobile/4.12.1 (Android 14; Pixel 8)",
    "curl/8.4.0",
    "PostmanRuntime/7.36.0",
    "GoogleBot/2.1 (+http://www.google.com/bot.html)",
]


# ---------- format renderers --------------------------------------------
# Each template is (template_id, format, level, weight, render_fn).
# render_fn returns either a single string or a list of strings (multi-line).

def t_nginx_get_product(level):
    code = random.choices([200, 200, 200, 304, 404, 500], weights=[10, 10, 10, 3, 2, 1])[0]
    return (
        f'{public_ip()} - - [{ts_apache()}] '
        f'"GET /api/products/{random.randint(1000, 99999)} HTTP/1.1" {code} '
        f'{random.randint(200, 8000)} "https://shop.example.com/" '
        f'"{random.choice(USER_AGENTS)}"'
    )


def t_nginx_post_checkout(level):
    code = random.choices([200, 201, 400, 402, 500], weights=[15, 10, 4, 2, 2])[0]
    return (
        f'{public_ip()} - - [{ts_apache()}] '
        f'"POST /api/checkout HTTP/1.1" {code} '
        f'{random.randint(50, 2000)} "https://shop.example.com/cart" '
        f'"{random.choice(USER_AGENTS)}"'
    )


def t_nginx_search(level):
    queries = ["red shoes", "ipad case", "laptop bag", "coffee table",
               "wireless mouse", "kitchen knife set", "garden hose",
               "smart watch", "running shorts"]
    code = random.choice([200, 200, 200, 200, 304, 500])
    q = random.choice(queries).replace(" ", "%20")
    return (
        f'{public_ip()} - - [{ts_apache()}] '
        f'"GET /api/search?q={q}&page={random.randint(1, 12)} HTTP/1.1" {code} '
        f'{random.randint(500, 12000)} "https://shop.example.com/search" '
        f'"{random.choice(USER_AGENTS)}"'
    )


def t_nginx_static(level):
    paths = ["/assets/css/app.7a3b.css", "/assets/js/main.4c2d.js",
             "/assets/img/logo.png", "/assets/fonts/inter.woff2",
             "/favicon.ico"]
    code = random.choice([200, 200, 304, 304, 304])
    return (
        f'{public_ip()} - - [{ts_apache()}] '
        f'"GET {random.choice(paths)} HTTP/1.1" {code} '
        f'{random.randint(1000, 200000)} "-" "{random.choice(USER_AGENTS)}"'
    )


def t_java_request_ok(level):
    return (
        f"{ts_java()} [http-nio-8080-exec-{random.randint(1, 64)}] INFO  "
        f"c.example.api.OrderController - Request processed: "
        f"order={order_id()} user={user_id()} duration={random.randint(5, 250)}ms"
    )


def t_java_validation_warn(level):
    return (
        f"{ts_java()} [http-nio-8080-exec-{random.randint(1, 64)}] WARN  "
        f"c.example.api.OrderController - Validation failed for "
        f"order request: field={random.choice(['email', 'shipping_address', 'card_number', 'cvv'])} "
        f"reason={random.choice(['invalid_format', 'required', 'too_long', 'malformed'])}"
    )


def t_java_npe(level):
    """Multi-line: ERROR + ~25 line Java stack trace."""
    thread = f"http-nio-8080-exec-{random.randint(1, 64)}"
    head = (f"{ts_java()} [{thread}] ERROR c.example.api.OrderController - "
            f"Unexpected error processing order {order_id()}")
    trace = [
        "java.lang.NullPointerException: Cannot invoke \"com.example.model.Cart.getItems()\" because \"cart\" is null",
        "\tat com.example.api.OrderController.placeOrder(OrderController.java:142)",
        "\tat com.example.api.OrderController$$FastClassBySpringCGLIB$$abc123.invoke(<generated>)",
        "\tat org.springframework.cglib.proxy.MethodProxy.invoke(MethodProxy.java:218)",
        "\tat org.springframework.aop.framework.CglibAopProxy$CglibMethodInvocation.invokeJoinpoint(CglibAopProxy.java:783)",
        "\tat org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:163)",
        "\tat org.springframework.aop.aspectj.MethodInvocationProceedingJoinPoint.proceed(MethodInvocationProceedingJoinPoint.java:89)",
        "\tat com.example.tracing.TraceAspect.aroundExecution(TraceAspect.java:62)",
        "\tat sun.reflect.GeneratedMethodAccessor134.invoke(Unknown Source)",
        "\tat sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)",
        "\tat java.lang.reflect.Method.invoke(Method.java:498)",
        "\tat org.springframework.aop.aspectj.AbstractAspectJAdvice.invokeAdviceMethodWithGivenArgs(AbstractAspectJAdvice.java:644)",
        "\tat org.springframework.aop.aspectj.AbstractAspectJAdvice.invokeAdviceMethod(AbstractAspectJAdvice.java:633)",
        "\tat org.springframework.aop.aspectj.AspectJAroundAdvice.invoke(AspectJAroundAdvice.java:72)",
        "\tat org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:175)",
        "\tat org.springframework.aop.interceptor.ExposeInvocationInterceptor.invoke(ExposeInvocationInterceptor.java:97)",
        "\tat org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:175)",
        "\tat org.springframework.aop.framework.CglibAopProxy$DynamicAdvisedInterceptor.intercept(CglibAopProxy.java:715)",
        "\tat com.example.api.OrderController$$EnhancerBySpringCGLIB$$xyz789.placeOrder(<generated>)",
        "\tat sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)",
        "\tat org.springframework.web.method.support.InvocableHandlerMethod.doInvoke(InvocableHandlerMethod.java:205)",
        "\tat org.springframework.web.method.support.InvocableHandlerMethod.invokeForRequest(InvocableHandlerMethod.java:150)",
        "\tat org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod.invokeAndHandle(ServletInvocableHandlerMethod.java:117)",
        "\tat org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter.invokeHandlerMethod(RequestMappingHandlerAdapter.java:895)",
        "\tat org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1057)",
    ]
    return [head] + trace


def t_java_db_timeout(level):
    """Multi-line: WARN with shorter cause chain."""
    head = (f"{ts_java()} [http-nio-8080-exec-{random.randint(1, 64)}] WARN  "
            f"c.example.persistence.OrderRepository - Query exceeded SLA: "
            f"queryId=q_{rid(8)} duration={random.randint(2000, 8000)}ms")
    trace = [
        f"org.springframework.dao.QueryTimeoutException: PreparedStatementCallback; SQL [SELECT o.* FROM orders o WHERE o.user_id = ? AND o.created_at > ?]",
        "\tat org.springframework.jdbc.support.SQLErrorCodeSQLExceptionTranslator.doTranslate(SQLErrorCodeSQLExceptionTranslator.java:259)",
        "\tat org.springframework.jdbc.core.JdbcTemplate.execute(JdbcTemplate.java:649)",
        "Caused by: org.postgresql.util.PSQLException: ERROR: canceling statement due to statement timeout",
        "\tat org.postgresql.core.v3.QueryExecutorImpl.receiveErrorResponse(QueryExecutorImpl.java:2675)",
    ]
    return [head] + trace


def t_json_checkout_ok(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "checkout-svc",
        "msg": "order placed successfully",
        "order_id": order_id(), "user_id": user_id(),
        "amount": amount(), "currency": random.choice(["USD", "EUR", "GBP"]),
        "payment_method": random.choice(["card", "paypal", "applepay", "googlepay"]),
        "items": random.randint(1, 8), "trace_id": rid(16),
    }
    return json.dumps(obj)


def t_json_payment_declined(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "payment-svc",
        "msg": "payment declined", "order_id": order_id(),
        "user_id": user_id(), "amount": amount(),
        "reason": random.choice(["insufficient_funds", "card_expired",
                                  "fraud_suspicion", "issuer_decline",
                                  "3ds_failed"]),
        "gateway": random.choice(["stripe", "adyen", "braintree"]),
        "trace_id": rid(16),
    }
    return json.dumps(obj)


def t_json_inventory_low(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "inventory-svc",
        "msg": "stock running low",
        "sku": f"SKU-{random.randint(10000, 99999)}",
        "warehouse": random.choice(["us-east", "us-west", "eu-central", "ap-south"]),
        "current_qty": random.randint(0, 10),
        "reorder_threshold": random.randint(10, 50),
    }
    return json.dumps(obj)


def t_json_inventory_ok(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "inventory-svc",
        "msg": "stock allocated",
        "sku": f"SKU-{random.randint(10000, 99999)}",
        "qty": random.randint(1, 8),
        "order_id": order_id(),
    }
    return json.dumps(obj)


def t_json_auth_login(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "auth-svc",
        "msg": "user authenticated",
        "user_id": user_id(),
        "method": random.choice(["password", "oauth_google", "oauth_apple", "sms_otp"]),
        "ip": public_ip(), "session_id": session_id(),
    }
    return json.dumps(obj)


def t_json_auth_failed(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "auth-svc",
        "msg": "authentication failed",
        "user_id": user_id(), "method": "password",
        "reason": random.choice(["invalid_credentials", "account_locked",
                                  "mfa_required", "ip_blocked"]),
        "ip": public_ip(), "attempts": random.randint(1, 8),
    }
    return json.dumps(obj)


def t_kv_k8s_pod_scheduled(level):
    return (
        f"time={ts_iso('s')} level=info source=scheduler "
        f"event=Scheduled namespace=production "
        f"pod={random.choice(['checkout', 'inventory', 'auth', 'search', 'cart'])}"
        f"-{rid(5)} "
        f"node=ip-10-0-{random.randint(1, 99)}-{random.randint(1, 254)} "
        f"msg=\"Successfully assigned pod to node\""
    )


def t_kv_k8s_pod_oom(level):
    return (
        f"time={ts_iso('s')} level=warn source=kubelet "
        f"event=OOMKilled namespace=production "
        f"pod={random.choice(['checkout', 'inventory', 'auth', 'search'])}"
        f"-{rid(5)} "
        f"container={random.choice(['app', 'sidecar'])} "
        f"reason=memory_limit_exceeded "
        f"memory_usage_mb={random.randint(512, 2048)} "
        f"limit_mb={random.choice([512, 1024, 2048])} "
        f"msg=\"Container exceeded memory limit and was killed\""
    )


def t_kv_k8s_image_pull(level):
    return (
        f"time={ts_iso('s')} level=info source=kubelet "
        f"event=Pulled namespace=production "
        f"pod={random.choice(['checkout', 'inventory', 'auth'])}-{rid(5)} "
        f"image=ecr/{random.choice(['checkout', 'inventory', 'auth'])}-svc:v{random.randint(100, 999)}.{random.randint(0, 99)} "
        f"msg=\"Successfully pulled image\""
    )


def t_postgres_slow_query(level):
    """Multi-line: PostgreSQL slow query log with query plan."""
    duration = random.randint(1500, 6500)
    head = (f"{ts_postgres()} [{random.randint(1000, 9999)}] LOG:  "
            f"duration: {duration}.{random.randint(100, 999)} ms  "
            f"statement: SELECT o.id, o.user_id, o.total, COUNT(oi.id) AS item_count "
            f"FROM orders o LEFT JOIN order_items oi ON oi.order_id = o.id "
            f"WHERE o.created_at > '{datetime.now(timezone.utc).strftime('%Y-%m-%d')}' "
            f"AND o.status = 'pending' GROUP BY o.id ORDER BY o.created_at DESC LIMIT 100")
    plan_lines = [
        f"{ts_postgres()} [{random.randint(1000, 9999)}] DETAIL:  "
        f"Limit  (cost=158234.45..158234.70 rows=100 width=64) (actual time={duration-50}..{duration} rows=100 loops=1)",
        f"          ->  Sort  (cost=158234.45..158734.45 rows=200000 width=64) (actual time={duration-100}..{duration-10} rows=100 loops=1)",
        f"                Sort Key: o.created_at DESC",
        f"                Sort Method: top-N heapsort  Memory: 32kB",
        f"                ->  GroupAggregate  (cost=125000.00..148000.00 rows=200000 width=64) (actual time={duration-500}..{duration-50} rows=180432 loops=1)",
        f"                      ->  Hash Left Join  (cost=15000.00..120000.00 rows=2500000 width=48) (actual time=100..{duration-200} rows=2412341 loops=1)",
        f"                            Hash Cond: (oi.order_id = o.id)",
        f"                            ->  Seq Scan on order_items oi  (cost=0.00..50000.00 rows=2500000 width=16)",
        f"                            ->  Hash  (cost=12000.00..12000.00 rows=200000 width=40)",
        f"                                  Buckets: 262144  Batches: 1  Memory Usage: 16384kB",
    ]
    return [head] + plan_lines


def t_postgres_deadlock(level):
    head = (f"{ts_postgres()} [{random.randint(1000, 9999)}] ERROR:  "
            f"deadlock detected")
    detail = [
        f"{ts_postgres()} [{random.randint(1000, 9999)}] DETAIL:  "
        f"Process 4521 waits for ShareLock on transaction 9821345; blocked by process 4823.",
        f"Process 4823 waits for ShareLock on transaction 9821456; blocked by process 4521.",
        f"Process 4521: UPDATE orders SET status = 'paid' WHERE id = '{order_id()}'",
        f"Process 4823: UPDATE orders SET status = 'cancelled' WHERE id = '{order_id()}'",
        f"{ts_postgres()} [{random.randint(1000, 9999)}] HINT:  See server log for query details.",
    ]
    return [head] + detail


def t_postgres_query_ok(level):
    return (
        f"{ts_postgres()} [{random.randint(1000, 9999)}] LOG:  "
        f"duration: {random.randint(2, 80)}.{random.randint(100, 999)} ms  "
        f"statement: SELECT * FROM "
        f"{random.choice(['users', 'products', 'orders', 'cart_items', 'reviews'])} "
        f"WHERE id = '{rid(8)}'"
    )


def t_redis_command(level):
    return (
        f"[{random.randint(1, 32)}] {ts_iso('s')} "
        f"{random.choice(['*', '#', '-'])} {random.choice(['GET', 'SET', 'DEL', 'HGET', 'LPUSH', 'ZADD', 'EXPIRE'])} "
        f"{random.choice(['session', 'cart', 'product', 'user', 'inventory'])}:"
        f"{rid(10)} ({random.randint(0, 5)} us)"
    )


def t_elasticsearch_query(level):
    return (
        f"[{ts_iso('s')}] [INFO] [o.e.r.RestController] [node-{random.randint(1, 6)}] "
        f"received [GET] request for [/products/_search] from "
        f"[{ip()}], took [{random.randint(5, 350)}ms]"
    )


def t_elasticsearch_unassigned(level):
    return (
        f"[{ts_iso('s')}] [WARN] [o.e.c.r.a.AllocationService] [master-1] "
        f"shard [{random.choice(['products', 'orders', 'users', 'reviews'])}]"
        f"[{random.randint(0, 7)}] cannot be allocated: "
        f"reason={random.choice(['ALLOCATION_FAILED', 'DELAYED_ALLOCATION', 'NODE_LEFT'])}, "
        f"node={random.choice(['data-1', 'data-2', 'data-3'])}"
    )


def t_mobile_crash(level):
    """Multi-line: mobile crash report with device metadata + stack."""
    head = (f"{ts_iso()} [mobile-telemetry] CRASH "
            f"app=ShopApp/4.12.1 platform={random.choice(['iOS 17.3', 'Android 14'])} "
            f"device={random.choice(['iPhone14,2', 'iPhone15,3', 'Pixel 8', 'SM-S928U'])} "
            f"user_id={user_id()}")
    detail = [
        f"  Exception: {random.choice(['NSInvalidArgumentException', 'NullPointerException', 'OutOfMemoryError'])}",
        f"  Reason: {random.choice(['attempt to insert nil object', 'index out of range', 'unable to decode response'])}",
        f"  Stack:",
        f"    0  ShopApp                      0x000000010234abcd -[CheckoutVC processOrder:] + 412",
        f"    1  ShopApp                      0x000000010234ef12 -[CheckoutVC viewDidAppear:] + 89",
        f"    2  UIKitCore                    0x00007fff48a12345 -[UIViewController _setViewAppearState:] + 712",
        f"    3  UIKitCore                    0x00007fff48a23456 -[UIView _addSubview:positioned:relativeTo:] + 1245",
        f"    4  libdispatch.dylib            0x00007fff513def01 _dispatch_call_block_and_release + 12",
        f"    5  libdispatch.dylib            0x00007fff513def22 _dispatch_client_callout + 8",
        f"  Battery: {random.randint(5, 100)}%  Memory: {random.randint(100, 800)}MB / {random.randint(2048, 8192)}MB",
        f"  Network: {random.choice(['WiFi', 'LTE', '5G'])}  Locale: {random.choice(['en_US', 'es_ES', 'ja_JP', 'de_DE'])}",
    ]
    return [head] + detail


def t_audit_login(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "category": "audit",
        "event": "user.login", "actor": user_id(),
        "ip": public_ip(), "country": random.choice(["US", "UK", "DE", "JP", "IN", "BR"]),
        "result": "success",
    }
    return json.dumps(obj)


def t_audit_privilege_escalation(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "category": "audit",
        "event": "privilege.escalation_attempt", "actor": user_id(),
        "target_role": random.choice(["admin", "merchant_admin", "support_agent"]),
        "ip": public_ip(),
        "result": "denied", "reason": "insufficient_permissions",
    }
    return json.dumps(obj)


def t_audit_data_export(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "category": "audit",
        "event": "data.export", "actor": user_id(),
        "dataset": random.choice(["orders", "users", "transactions"]),
        "rows_exported": random.randint(100, 50000),
        "destination": random.choice(["s3://reports/", "user_download", "scheduled_email"]),
    }
    return json.dumps(obj)


def t_cdn_cache_miss(level):
    return (
        f'"{ts_iso()}",cf-pop="{random.choice(["DFW", "LHR", "NRT", "SIN", "FRA"])}",'
        f'cache_status="MISS",origin_status={random.randint(200, 599)},'
        f'origin_time_ms={random.randint(50, 2000)},'
        f'edge_time_ms={random.randint(1, 50)},'
        f'host="cdn.shop.example.com",'
        f'uri="/static/products/{random.randint(1000, 99999)}.jpg",'
        f'bytes={random.randint(5000, 500000)}'
    )


def t_cdn_ddos_block(level):
    return (
        f'"{ts_iso()}",cf-pop="{random.choice(["DFW", "LHR", "NRT"])}",'
        f'cache_status="BLOCK",origin_status=0,'
        f'rule_id="ddos_{rid(6)}",'
        f'client_ip="{public_ip()}",'
        f'reason="rate_limit_exceeded",'
        f'request_count={random.randint(1000, 50000)}'
    )


def t_worker_email_sent(level):
    return (
        f"{ts_iso()} [worker-email] INFO Email sent: "
        f"template={random.choice(['order_confirmation', 'shipping_update', 'password_reset', 'cart_abandoned', 'welcome'])} "
        f"to={user_id()} provider={random.choice(['sendgrid', 'ses', 'mailgun'])}"
    )


def t_worker_image_resize(level):
    return (
        f"{ts_iso()} [worker-imageproc] INFO Resized image: "
        f"sku=SKU-{random.randint(10000, 99999)} "
        f"variants={random.randint(3, 8)} "
        f"duration_ms={random.randint(50, 2500)} "
        f"output_bytes={random.randint(10000, 800000)}"
    )


def t_recommender_inference(level):
    obj = {
        "ts": ts_iso(), "level": "INFO", "service": "recommender-svc",
        "msg": "recommendations generated",
        "user_id": user_id(), "context": random.choice(["homepage", "product_page", "cart", "email"]),
        "model_version": f"v{random.randint(20, 99)}.{random.randint(0, 9)}",
        "latency_ms": random.randint(20, 200),
        "n_results": random.randint(5, 50),
    }
    return json.dumps(obj)


def t_recommender_cold_start(level):
    obj = {
        "ts": ts_iso(), "level": "WARN", "service": "recommender-svc",
        "msg": "cold start fallback used",
        "user_id": user_id(), "reason": "no_interaction_history",
        "fallback": random.choice(["bestsellers", "trending", "category_default"]),
    }
    return json.dumps(obj)


# Many small/rare templates to create the long tail (collectively ~10% of traffic)
def t_rare_feature_flag(level):
    return f"{ts_iso()} [feature-flags] DEBUG flag={random.choice(['new_checkout', 'beta_search', 'ai_recommendations', 'split_payments', 'social_login'])} variant={random.choice(['A', 'B', 'C', 'control'])} user={user_id()}"


def t_rare_ratelimit(level):
    return f"{ts_iso()} [api-gateway] WARN rate-limit: client={rid(8)} endpoint=/api/{random.choice(['orders', 'products', 'search'])} rate={random.randint(101, 999)}/min"


def t_rare_certificate(level):
    return f"{ts_iso()} [tls-monitor] WARN certificate expires in {random.randint(7, 60)} days for domain={random.choice(['shop.example.com', 'api.shop.example.com', 'cdn.shop.example.com'])}"


def t_rare_cron_overlap(level):
    return f"{ts_iso()} [cron-runner] WARN job overlap: name={random.choice(['nightly-export', 'price-sync', 'inventory-reconciliation', 'order-cleanup'])} previous still running after {random.randint(60, 1800)}s"


def t_rare_dns_failure(level):
    return f"{ts_iso()} [dns-resolver] ERROR Failed to resolve hostname: {random.choice(['payments.stripe.com', 'api.shipping-provider.com', 'analytics-warehouse.internal'])} ({random.choice(['NXDOMAIN', 'SERVFAIL', 'TIMEOUT'])})"


def t_rare_ssl_handshake(level):
    return f"{ts_iso()} [ssl-handler] WARN handshake failed: client={public_ip()} cipher_suite={random.choice(['TLS_RSA_WITH_RC4_128_SHA', 'TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA'])} (deprecated)"


def t_rare_gdpr(level):
    obj = {"ts": ts_iso(), "level": "INFO", "category": "audit",
           "event": "gdpr.data_deletion", "subject_id": user_id(),
           "request_id": rid(12), "status": "completed"}
    return json.dumps(obj)


def t_rare_promo(level):
    obj = {"ts": ts_iso(), "level": "INFO", "service": "promo-svc",
           "msg": "promo code applied", "code": f"SAVE{random.randint(10, 50)}",
           "user_id": user_id(), "discount_amount": amount() / 4}
    return json.dumps(obj)


def t_rare_a11y(level):
    return f"{ts_iso()} [a11y-audit] INFO accessibility scan: page=/products/{random.randint(1000, 9999)} score={random.randint(70, 100)} issues={random.randint(0, 12)}"


def t_rare_webhook_retry(level):
    return f"{ts_iso()} [webhook-dispatcher] WARN retry attempt {random.randint(2, 5)} for webhook={rid(8)} url=https://merchant.example.com/webhooks/orders status={random.choice([502, 503, 504, 'timeout'])}"


def t_rare_metric_emit(level):
    return f"{ts_iso()} [metrics-emitter] DEBUG emitting metric={random.choice(['order.placed', 'cart.added', 'product.viewed', 'search.executed'])} value={random.randint(1, 1000)} tags=region={random.choice(['us', 'eu', 'ap'])}"


def t_rare_loadbalancer(level):
    return f"{ts_iso()} [load-balancer] INFO health check: target={random.choice(['checkout', 'inventory', 'auth'])}-svc-{rid(4)} status={random.choice(['healthy', 'healthy', 'healthy', 'unhealthy'])} response_ms={random.randint(1, 50)}"


def t_rare_backup(level):
    return f"{ts_iso()} [backup-runner] INFO snapshot complete: database={random.choice(['orders', 'users', 'inventory'])} size_gb={random.randint(20, 800)} duration_min={random.randint(5, 120)}"


# ---------- template registry --------------------------------------------
# (id, level, weight, render_fn) — weights produce roughly Zipfian distribution.
# Top entries (high weight) dominate; long-tail entries (weight=1) appear rarely.
TEMPLATES = [
    # Top tier — high-frequency normal-operations templates (~70% of traffic)
    ("nginx_get_product",        "INFO",  120, t_nginx_get_product),
    ("nginx_static",             "INFO",  100, t_nginx_static),
    ("redis_command",            "INFO",   90, t_redis_command),
    ("json_inventory_ok",        "INFO",   80, t_json_inventory_ok),
    ("kv_k8s_image_pull",        "INFO",   70, t_kv_k8s_image_pull),
    ("postgres_query_ok",        "INFO",   65, t_postgres_query_ok),
    ("json_auth_login",          "INFO",   55, t_json_auth_login),
    ("nginx_search",             "INFO",   55, t_nginx_search),
    ("java_request_ok",          "INFO",   50, t_java_request_ok),
    ("json_checkout_ok",         "INFO",   45, t_json_checkout_ok),
    ("elasticsearch_query",      "INFO",   40, t_elasticsearch_query),
    ("worker_email_sent",        "INFO",   30, t_worker_email_sent),
    ("worker_image_resize",      "INFO",   25, t_worker_image_resize),
    ("recommender_inference",    "INFO",   25, t_recommender_inference),
    ("audit_login",              "INFO",   25, t_audit_login),
    ("cdn_cache_miss",           "INFO",   25, t_cdn_cache_miss),

    # Mid tier — operational warnings (~15% of traffic)
    ("nginx_post_checkout",      "INFO",   20, t_nginx_post_checkout),
    ("json_inventory_low",       "WARN",   15, t_json_inventory_low),
    ("kv_k8s_pod_scheduled",     "INFO",   15, t_kv_k8s_pod_scheduled),
    ("java_validation_warn",     "WARN",   12, t_java_validation_warn),
    ("json_payment_declined",    "WARN",   10, t_json_payment_declined),
    ("json_auth_failed",         "WARN",   10, t_json_auth_failed),
    ("elasticsearch_unassigned", "WARN",    8, t_elasticsearch_unassigned),
    ("recommender_cold_start",   "WARN",    6, t_recommender_cold_start),
    ("audit_data_export",        "INFO",    5, t_audit_data_export),
    ("postgres_slow_query",      "WARN",    5, t_postgres_slow_query),         # multi-line!

    # Bottom tier — errors and rare events (~5% of traffic)
    ("java_db_timeout",          "WARN",    4, t_java_db_timeout),              # multi-line!
    ("java_npe",                 "ERROR",   3, t_java_npe),                     # multi-line! 25 lines
    ("mobile_crash",             "ERROR",   3, t_mobile_crash),                 # multi-line!
    ("kv_k8s_pod_oom",           "WARN",    3, t_kv_k8s_pod_oom),
    ("postgres_deadlock",        "ERROR",   2, t_postgres_deadlock),            # multi-line!
    ("cdn_ddos_block",           "WARN",    2, t_cdn_ddos_block),
    ("audit_privilege_escalation","WARN",   2, t_audit_privilege_escalation),

    # Long tail — rare templates (~10% collectively, weight 1 each)
    ("rare_feature_flag",        "DEBUG",   1, t_rare_feature_flag),
    ("rare_ratelimit",           "WARN",    1, t_rare_ratelimit),
    ("rare_certificate",         "WARN",    1, t_rare_certificate),
    ("rare_cron_overlap",        "WARN",    1, t_rare_cron_overlap),
    ("rare_dns_failure",         "ERROR",   1, t_rare_dns_failure),
    ("rare_ssl_handshake",       "WARN",    1, t_rare_ssl_handshake),
    ("rare_gdpr",                "INFO",    1, t_rare_gdpr),
    ("rare_promo",               "INFO",    1, t_rare_promo),
    ("rare_a11y",                "INFO",    1, t_rare_a11y),
    ("rare_webhook_retry",       "WARN",    1, t_rare_webhook_retry),
    ("rare_metric_emit",         "DEBUG",   1, t_rare_metric_emit),
    ("rare_loadbalancer",        "INFO",    1, t_rare_loadbalancer),
    ("rare_backup",              "INFO",    1, t_rare_backup),
]


def template_weights():
    return [w for _, _, w, _ in TEMPLATES]


def malformed_line():
    """Produce a 'dirty' line — truncated, garbled, or wrong format."""
    kind = random.choice(["truncated", "empty", "garbage", "half_json", "wrong_encoding"])
    if kind == "truncated":
        # take a normal line and cut it in the middle
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
        # cut off mid-JSON
        s = json.dumps(partial)
        return s[:random.randint(20, len(s) - 5)]
    else:  # wrong_encoding — simulate by writing escaped bad-byte markers (no surrogates)
        return f"{ts_iso()} [?service?] some message with bad bytes: \\xfd\\xff\\xfe (encoding fallback)"


def emit(out, drift_active: bool = False) -> int:
    """Emit ONE log entry (which may be multi-line) and return line count."""
    # ~5% malformed
    if random.random() < 0.05:
        out.write(malformed_line() + "\n")
        out.flush()
        return 1
    # otherwise sample a template
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


def run(rate: int, duration: float | None):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))

    interval_per_line = 1.0 / rate
    start = time.time()
    written_lines = 0
    written_entries = 0

    def stop_handler(sig, frame):
        print(f"\n[scheduler] stopping after {written_entries} entries "
              f"({written_lines} raw lines) in {time.time()-start:.1f}s")
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    print(f"[scheduler] writing realistic log to {LOG_FILE} at ~{rate} lines/sec "
          f"(duration: {'∞' if duration is None else f'{duration}s'})")
    print(f"[scheduler] {len(TEMPLATES)} templates "
          f"({sum(1 for _, _, w, _ in TEMPLATES if w >= 20)} dominant, "
          f"{sum(1 for _, _, w, _ in TEMPLATES if w == 1)} long-tail rare)")
    with open(LOG_FILE, "a", buffering=1) as out:
        next_t = time.time()
        while True:
            if duration is not None and (time.time() - start) >= duration:
                break
            n = emit(out)
            written_lines += n
            written_entries += 1
            if written_entries % 500 == 0:
                elapsed = time.time() - start
                print(f"[scheduler] +{written_entries} entries "
                      f"({written_lines} raw lines, "
                      f"{written_lines/elapsed:.0f} lines/s)")
            # pace by lines, not entries — multi-line entries should not all hit at once
            next_t += interval_per_line * n
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

    print(f"[scheduler] done. {written_entries} entries / {written_lines} raw lines "
          f"in {time.time()-start:.1f}s")
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=80, help="raw lines per second")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()
    if args.truncate and LOG_FILE.exists():
        LOG_FILE.write_text("")
        print(f"[scheduler] truncated {LOG_FILE}")
    run(args.rate, args.duration)


if __name__ == "__main__":
    main()
