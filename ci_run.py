#!/usr/bin/env python3
"""
GitHub Actions runner(windows-latest)에서 실행되는 CI 진입점.

- RUN_ID 환경변수가 있으면: hurotics-sales 대시보드에서 manifest(GET)를 받아
  exe + 데이터셋을 signed URL로 다운로드하고 corpus 폴더를 재구성한 뒤
  runcorpus.py를 실행, 결과를 finish 엔드포인트(POST)로 push한다.
- RUN_ID가 없으면(plain push 트리거): test_corpus 스모크 테스트로 폴백한다.
  단 test_corpus/와 exe는 repo에 없으므로(.gitignore) CI 체크아웃에서는
  아무것도 하지 않고 0으로 끝난다 — 로컬에 파일이 있을 때만 실제로 돈다.

대시보드 쪽 wire-format 계약: hurotics-sales repo의 lib/algo/run-contract.ts 참고.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

RUN_ID = os.environ.get("RUN_ID", "").strip()
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").rstrip("/")
SECRET = os.environ.get("ALGO_INGEST_SECRET", "")
# Vercel Preview 배포는 기본 SSO 보호벽 뒤에 있음 — 자동화(CI) 우회용 시크릿.
# Vercel 프로젝트 Settings > Deployment Protection > Protection Bypass for Automation에서 발급.
VERCEL_BYPASS = os.environ.get("VERCEL_PROTECTION_BYPASS_SECRET", "")


def dashboard_headers(extra: dict | None = None) -> dict:
    headers = {"x-algo-secret": SECRET}
    if VERCEL_BYPASS:
        headers["x-vercel-protection-bypass"] = VERCEL_BYPASS
    if extra:
        headers.update(extra)
    return headers


def fallback_smoke_test() -> int:
    # test_corpus/ (실제 시행 데이터)와 exe는 더 이상 repo에 커밋하지 않는다 — .gitignore 참고.
    # 로컬에 있으면 스모크 테스트를 돌리고, 없으면 (= CI 체크아웃) 조용히 건너뛴다.
    corpus, exe = Path("test_corpus"), Path("loco_classifier_app.exe")
    missing = [str(p) for p in (corpus, exe) if not p.exists()]
    if missing:
        print(f"[ci_run] smoke test skipped — not in this checkout: {', '.join(missing)}")
        print("[ci_run] 실제 평가를 돌리려면 workflow_dispatch로 run_id를 넘기세요.")
        return 0

    return subprocess.run([
        "python", "runcorpus.py", "test_corpus", "test_corpus", "loco_classifier_app.exe",
        "--json", "report_technical.json",
        "--json-nontechnical", "report_nontechnical.json",
    ]).returncode


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=dashboard_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def download_gunzip(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".gz")
    download(url, tmp)
    with gzip.open(tmp, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    tmp.unlink()


def push_finish(payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{DASHBOARD_URL}/api/algo/runs/{RUN_ID}/finish",
        data=body,
        method="POST",
        headers=dashboard_headers({"content-type": "application/json"}),
    )
    try:
        urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"[ci_run] finish push HTTP {e.code}: {detail}", file=sys.stderr)
        raise


def trial_report_path(report_dir: Path, group_id: str, trial_id: str) -> Path:
    # locoeval/report.py write_reports()와 동일 규칙:
    # report_dir/<group_id>/report_nontechnical_<basename(trial_id) 확장자 제외>.json
    basename = Path(trial_id).stem
    d = (report_dir / group_id) if group_id else report_dir
    return d / f"report_nontechnical_{basename}.json"


def main() -> int:
    if not RUN_ID:
        return fallback_smoke_test()

    if not DASHBOARD_URL or not SECRET:
        print("[ci_run] DASHBOARD_URL / ALGO_INGEST_SECRET 환경변수가 설정되지 않았어요.", file=sys.stderr)
        return 2

    manifest = fetch_json(f"{DASHBOARD_URL}/api/algo/runs/{RUN_ID}/manifest")

    exe_path = Path("run_exe.exe")
    download(manifest["exe"]["signed_url"], exe_path)

    input_root = Path("corpus_input")
    gt_root = Path("corpus_gt")
    for root in (input_root, gt_root):
        if root.exists():
            shutil.rmtree(root)

    for ds in manifest["datasets"]:
        trial_id = ds["trial_id"]  # input_root 기준 상대경로 그대로 사용
        download_gunzip(ds["input_signed_url"], input_root / trial_id)
        download_gunzip(ds["ground_truth_signed_url"], gt_root / trial_id)

    report_dir = Path("reports")
    cmd = [
        "python", "runcorpus.py", str(input_root), str(gt_root), str(exe_path),
        "--json", "report_technical.json",
        "--json-nontechnical", "report_nontechnical.json",
        "--report-dir", str(report_dir),
        "--quiet",
    ]
    late_ms = manifest.get("late_threshold_ms")
    if late_ms:
        cmd += ["--late-threshold-ms", str(late_ms)]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        push_finish({"ok": False, "error": f"runcorpus.py exited with code {result.returncode}"})
        return result.returncode

    technical = json.loads(Path("report_technical.json").read_text(encoding="utf-8"))
    nontechnical = json.loads(Path("report_nontechnical.json").read_text(encoding="utf-8"))

    trials = []
    for ds in manifest["datasets"]:
        p = trial_report_path(report_dir, ds["group_id"], ds["trial_id"])
        if not p.exists():
            continue  # 이 trial은 실패/스킵됨 — trial_results 없이 넘어감
        trial_report = json.loads(p.read_text(encoding="utf-8"))
        trials.append({
            "group_id": ds["group_id"],
            "trial_id": ds["trial_id"],
            "frame_error_rate": trial_report.get("frame_error_rate"),
            "bucket_counts": {
                "transition_pairs": trial_report.get("transition_pairs"),
                "false_transitions": trial_report.get("false_transitions"),
                "steady_state_errors": trial_report.get("steady_state_errors"),
            },
        })

    push_finish({
        "ok": True,
        "technical_report": technical,
        "nontechnical_report": nontechnical,
        "trials": trials,
    })
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # 최후 방어선 — 어떤 예외든 finish로 실패 보고 후 종료 (run이 running에 멈춰있지 않게)
        print(f"[ci_run] unhandled exception: {e}", file=sys.stderr)
        if RUN_ID and DASHBOARD_URL and SECRET:
            try:
                push_finish({"ok": False, "error": f"ci_run.py unhandled exception: {e}"})
            except Exception:
                pass
        sys.exit(1)
