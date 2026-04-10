import os
import re
import cv2
import math
import json
import wave
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np


# ============================================================
# Video Highlight AI - All-in-One Version
# ============================================================
# 포함 기능
# 1) 하이라이트 자동 추출
#    - 오디오 에너지
#    - 움직임
#    - 장면 전환
# 2) 대사/자막 기반 점수 보정 (Whisper 사용 가능)
# 3) GUI 앱 (Streamlit)
# 4) 유튜브 쇼츠용 세로 영상 생성 (9:16)
# 5) exe 빌드용 가이드 포함
#
# ------------------------------------------------------------
# 설치
# pip install opencv-python numpy streamlit
#
# 선택 설치 (대사 분석)
# pip i([github.com](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file&utm_source=chatgpt.com))영상 다운로드 기능
# pip install yt-dlp
#
# ffmpeg 설치 후 PATH 등록 필요
# 확인:
#   ffmpeg -version
#   ffprobe -version
#
# ------------------------------------------------------------
# 실행 방법
# 1) GUI 실행
#    streamlit run video_highlight_ai_all_in_one.py
#
# 2) 콘솔 실행
#    python video_highlight_ai_all_in_one.py --input input.mp4 --output highlight.mp4
#
# 3) 쇼츠 생성 포함
#    python video_highlight_ai_all_in_one.py --input input.mp4 --output highlight.mp4 --make-shorts
#
# 4) 자막 분석 포함
#    python video_highlight_ai_all_in_one.py --input input.mp4 --output highlight.mp4 --use-whisper
#
# 5) exe 만들기
#    pip install pyinstaller
#    pyinstaller --onefile video_highlight_ai_all_in_one.py
#
# GUI exe 예시 (streamlit 앱은 별도 실행형으로 묶기 까다로워서
# 실사용은 콘솔 엔진 exe + bat 실행 방식 추천)
#
# ------------------------------------------------------------
# 파일 구조 예시
#   project/
#     video_highlight_ai_all_in_one.py
#     input.mp4
#     outputs/
#
# ------------------------------------------------------------
# 참고
# - Whisper는 CPU에서 느릴 수 있음
# - faster-whisper 설치가 안 되면 자동으로 비활성화됨
# - 긴 영상은 처리 시간이 걸릴 수 있음
# ============================================================


# =========================
# Data Classes
# =========================
@dataclass
class Segment:
    start: float
    end: float
    score: float
    reason: str = ""


@dataclass
class WhisperLine:
    start: float
    end: float
    text: str
    score: float


@dataclass
class Config:
    input_path: str
    output_path: str
    target_seconds: int = 60
    window_seconds: float = 1.0
    clip_min: float = 2.0
    clip_max: float = 8.0
    merge_gap: float = 1.2
    sample_width: int = 320
    audio_weight: float = 0.30
    motion_weight: float = 0.30
    scene_weight: float = 0.20
    speech_weight: float = 0.20
    use_whisper: bool = False
    whisper_model: str = "small"
    make_shorts: bool = False
    shorts_count: int = 3
    shorts_duration: int = 20
    shorts_width: int = 1080
    shorts_height: int = 1920


# =========================
# Utility Functions
# =========================
def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"명령 실행 실패:\n{' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return result



def ffprobe_duration(path: str) -> float:
    result = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    return float(result.stdout.strip())



def normalize_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)



def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)



def sec_to_hhmmss(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', '_', name)
    return name.strip() or 'downloaded_video'


def download_video_from_url(video_url: str, output_dir: str) -> str:
    try:
        import yt_dlp
    except Exception as e:
        raise RuntimeError("yt-dlp가 설치되지 않았습니다. pip install yt-dlp 후 다시 시도하세요.") from e

    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, '%(title).120s.%(ext)s')

    ydl_opts = {
        'outtmpl': outtmpl,
        'format': 'mp4/bv*+ba/b',
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        if info is None:
            raise RuntimeError('URL에서 영상 정보를 가져오지 못했습니다.')

        if 'entries' in info and info['entries']:
            info = info['entries'][0]

        downloaded_path = ydl.prepare_filename(info)
        base_no_ext = os.path.splitext(downloaded_path)[0]
        merged_mp4 = base_no_ext + '.mp4'
        if os.path.exists(merged_mp4):
            return merged_mp4
        if os.path.exists(downloaded_path):
            return downloaded_path

        title = sanitize_filename(info.get('title', 'downloaded_video'))
        for file_name in os.listdir(output_dir):
            if file_name.startswith(title):
                return os.path.join(output_dir, file_name)

    raise RuntimeError('영상 다운로드는 완료됐지만 결과 파일을 찾지 못했습니다.')


# =========================
# Audio Analysis
# =========================
def extract_audio_wav(input_path: str, wav_path: str) -> None:
    run_cmd([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-vn",
        wav_path,
    ])



def read_wav_pcm16_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        if channels != 1 or sampwidth != 2:
            raise ValueError("WAV 파일이 mono 16-bit PCM 형식이 아닙니다.")
        raw = wf.readframes(nframes)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, framerate



def audio_energy_per_window(input_path: str, duration: float, window_sec: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        extract_audio_wav(input_path, wav_path)
        audio, sr = read_wav_pcm16_mono(wav_path)

    win = max(1, int(sr * window_sec))
    n = int(math.ceil(duration / window_sec))
    energies = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s = i * win
        e = min(len(audio), (i + 1) * win)
        if s >= len(audio):
            break
        chunk = audio[s:e]
        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-8))
        energies[i] = rms

    return normalize_array(energies)


# =========================
# Video Analysis
# =========================
def analyze_video_windows(input_path: str, duration: float, window_sec: float, sample_width: int) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 30.0

    total_windows = int(math.ceil(duration / window_sec))
    motion_scores = np.zeros(total_windows, dtype=np.float32)
    scene_scores = np.zeros(total_windows, dtype=np.float32)
    counts = np.zeros(total_windows, dtype=np.float32)

    prev_gray = None
    prev_hist = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        scale = sample_width / float(w)
        resized = cv2.resize(frame, (sample_width, max(1, int(h * scale))))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        current_time = frame_idx / fps
        win_idx = min(total_windows - 1, int(current_time // window_sec))

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores[win_idx] += float(np.mean(diff))

            if prev_hist is not None:
                scene = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                scene_scores[win_idx] += float(scene)

        prev_gray = gray
        prev_hist = hist
        counts[win_idx] += 1.0
        frame_idx += 1

    cap.release()

    valid = counts > 0
    motion_scores[valid] /= counts[valid]
    scene_scores[valid] /= counts[valid]

    return normalize_array(motion_scores), normalize_array(scene_scores)


# =========================
# Speech / Subtitle Analysis
# =========================
def clean_text_score(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0

    score = 0.0

    # 글자 수가 너무 짧지 않으면 가산점
    score += min(len(text) / 30.0, 1.0) * 0.35

    # 감탄/의문/강조 표현
    if re.search(r"[!?！？]", text):
        score += 0.20

    # 숫자/핵심 정보성 문장
    if re.search(r"\d", text):
        score += 0.10

    # 강조 키워드 예시
    keywords = [
        "와", "대박", "진짜", "미쳤", "레전드", "중요", "핵심", "결정", "우승", "골",
        "킬", "성공", "실패", "역전", "최고", "끝", "바로", "드디어", "why", "wow",
        "amazing", "insane", "goal", "winner", "important", "final"
    ]
    lowered = text.lower()
    if any(k in lowered for k in keywords):
        score += 0.35

    return min(score, 1.0)



def transcribe_with_whisper(input_path: str, model_name: str = "small") -> List[WhisperLine]:
    try:
        from faster_whisper import WhisperModel
    except Exception:
        print("[WARN] faster-whisper가 설치되지 않아 대사 분석을 건너뜁니다.")
        return []

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(input_path, vad_filter=True)

    results: List[WhisperLine] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        score = clean_text_score(text)
        results.append(WhisperLine(
            start=float(seg.start),
            end=float(seg.end),
            text=text,
            score=score,
        ))
    return results



def speech_score_per_window(lines: List[WhisperLine], duration: float, window_sec: float) -> np.ndarray:
    n = int(math.ceil(duration / window_sec))
    scores = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    for line in lines:
        s = int(max(0.0, line.start) // window_sec)
        e = int(min(duration, line.end) // window_sec)
        for i in range(s, min(n, e + 1)):
            scores[i] += line.score
            counts[i] += 1.0

    valid = counts > 0
    scores[valid] /= counts[valid]
    return normalize_array(scores)


# =========================
# Highlight Scoring
# =========================
def score_windows(audio: np.ndarray, motion: np.ndarray, scene: np.ndarray, speech: np.ndarray, cfg: Config) -> np.ndarray:
    n = min(len(audio), len(motion), len(scene), len(speech))
    score = (
        cfg.audio_weight * audio[:n] +
        cfg.motion_weight * motion[:n] +
        cfg.scene_weight * scene[:n] +
        cfg.speech_weight * speech[:n]
    )
    return normalize_array(score)



def merge_segments(segments: List[Segment], gap: float) -> List[Segment]:
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start - last.end <= gap:
            merged[-1] = Segment(
                start=last.start,
                end=max(last.end, seg.end),
                score=max(last.score, seg.score),
                reason=(last.reason + " | " + seg.reason).strip(" |"),
            )
        else:
            merged.append(seg)
    return merged



def pick_segments(scores: np.ndarray, cfg: Config, duration: float, speech_lines: Optional[List[WhisperLine]] = None) -> List[Segment]:
    n = len(scores)
    order = list(np.argsort(scores)[::-1])
    selected: List[Segment] = []
    used = np.zeros(n, dtype=bool)
    total = 0.0
    base_half = max(cfg.clip_min / 2.0, 1.0)

    speech_lines = speech_lines or []

    for idx in order:
        if total >= cfg.target_seconds:
            break
        if used[idx]:
            continue

        strength = float(scores[idx])
        clip_len = cfg.clip_min + (cfg.clip_max - cfg.clip_min) * strength
        clip_len = max(cfg.clip_min, min(cfg.clip_max, clip_len))

        start = max(0.0, idx * cfg.window_seconds - base_half)
        end = min(duration, start + clip_len)
        start = max(0.0, end - clip_len)

        s_idx = int(start // cfg.window_seconds)
        e_idx = min(n - 1, int(end // cfg.window_seconds))
        overlap = used[s_idx:e_idx + 1].mean() if e_idx >= s_idx else 0.0
        if overlap > 0.45:
            continue

        reason_texts = []
        for line in speech_lines:
            if line.end < start or line.start > end:
                continue
            if line.score >= 0.45:
                reason_texts.append(line.text)
                if len(reason_texts) >= 2:
                    break

        reason = " / ".join(reason_texts[:2])

        used[s_idx:e_idx + 1] = True
        selected.append(Segment(start=start, end=end, score=strength, reason=reason))
        total += (end - start)

    selected.sort(key=lambda x: x.start)
    return merge_segments(selected, cfg.merge_gap)


# =========================
# Rendering / Export
# =========================
def write_concat_parts(input_path: str, segments: List[Segment], temp_dir: str) -> List[str]:
    part_files = []
    for i, seg in enumerate(segments):
        part_path = os.path.join(temp_dir, f"part_{i:03d}.mp4")
        run_cmd([
            "ffmpeg", "-y",
            "-ss", f"{seg.start:.3f}",
            "-to", f"{seg.end:.3f}",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "20",
            "-c:a", "aac",
            "-movflags", "+faststart",
            part_path,
        ])
        part_files.append(part_path)
    return part_files



def concat_parts(part_files: List[str], output_path: str, temp_dir: str) -> None:
    list_path = os.path.join(temp_dir, "concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in part_files:
            safe_p = p.replace("'", "'\\''")
            f.write(f"file '{safe_p}'\n")

    run_cmd([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ])



def create_highlight_video(input_path: str, output_path: str, segments: List[Segment]) -> None:
    ensure_parent_dir(output_path)
    with tempfile.TemporaryDirectory() as td:
        part_files = write_concat_parts(input_path, segments, td)
        concat_parts(part_files, output_path, td)



def create_shorts_from_segments(input_path: str, output_dir: str, segments: List[Segment], shorts_count: int, shorts_duration: int, width: int, height: int) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    outputs = []

    top_segments = sorted(segments, key=lambda s: s.score, reverse=True)[:shorts_count]
    for i, seg in enumerate(top_segments, 1):
        center_start = seg.start
        center_end = min(seg.end, seg.start + shorts_duration)
        if center_end - center_start < 3:
            center_end = min(seg.start + max(shorts_duration, 6), seg.end)

        out_path = os.path.join(output_dir, f"shorts_{i:02d}.mp4")

        vf = (
            f"scale=-2:{height},"
            f"crop={width}:{height},"
            f"setsar=1"
        )

        run_cmd([
            "ffmpeg", "-y",
            "-ss", f"{center_start:.3f}",
            "-to", f"{center_end:.3f}",
            "-i", input_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-c:a", "aac",
            out_path,
        ])
        outputs.append(out_path)

    return outputs



def save_report(output_path: str, cfg: Config, segments: List[Segment], speech_lines: List[WhisperLine]) -> str:
    report_path = os.path.splitext(output_path)[0] + "_report.json"
    data = {
        "config": asdict(cfg),
        "segments": [asdict(s) for s in segments],
        "speech_lines": [asdict(x) for x in speech_lines[:300]],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return report_path


# =========================
# Main Engine
# =========================
def build_highlight(cfg: Config) -> dict:
    if not os.path.exists(cfg.input_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {cfg.input_path}")

    duration = ffprobe_duration(cfg.input_path)
    print(f"[1/6] 영상 길이: {duration:.2f}초")

    print("[2/6] 오디오 분석 중...")
    audio_scores = audio_energy_per_window(cfg.input_path, duration, cfg.window_seconds)

    print("[3/6] 프레임 분석 중...")
    motion_scores, scene_scores = analyze_video_windows(
        cfg.input_path,
        duration,
        cfg.window_seconds,
        cfg.sample_width,
    )

    print("[4/6] 대사 분석 중...")
    speech_lines: List[WhisperLine] = []
    if cfg.use_whisper:
        speech_lines = transcribe_with_whisper(cfg.input_path, cfg.whisper_model)
    speech_scores = speech_score_per_window(speech_lines, duration, cfg.window_seconds)

    print("[5/6] 하이라이트 점수 계산 중...")
    scores = score_windows(audio_scores, motion_scores, scene_scores, speech_scores, cfg)
    segments = pick_segments(scores, cfg, duration, speech_lines)
    if not segments:
        raise RuntimeError("하이라이트 구간을 찾지 못했습니다. 설정값을 조정해 보세요.")

    print("선택된 구간:")
    for i, seg in enumerate(segments, 1):
        print(f"  {i}. {seg.start:.2f}s ~ {seg.end:.2f}s | score={seg.score:.3f} | {seg.reason[:60]}")

    print("[6/6] 결과 영상 생성 중...")
    create_highlight_video(cfg.input_path, cfg.output_path, segments)
    report_path = save_report(cfg.output_path, cfg, segments, speech_lines)

    shorts_outputs = []
    if cfg.make_shorts:
        shorts_dir = os.path.join(os.path.dirname(cfg.output_path) or ".", "shorts_outputs")
        shorts_outputs = create_shorts_from_segments(
            cfg.input_path,
            shorts_dir,
            segments,
            cfg.shorts_count,
            cfg.shorts_duration,
            cfg.shorts_width,
            cfg.shorts_height,
        )

    return {
        "output_video": cfg.output_path,
        "report_path": report_path,
        "segments": segments,
        "shorts_outputs": shorts_outputs,
        "speech_lines_count": len(speech_lines),
    }


# =========================
# Streamlit GUI
# =========================
def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Video Highlight AI", layout="wide")
    st.title("🎬 Video Highlight AI")
    st.caption("하이라이트 추출 + 자막 분석 + 쇼츠 생성 + exe용 엔진")

    source_type = st.radio("입력 방식", ["파일 업로드", "영상 URL"], horizontal=True)

    uploaded = None
    video_url = ""
    if source_type == "파일 업로드":
        uploaded = st.file_uploader("영상 업로드", type=["mp4", "mov", "avi", "mkv"])
    else:
        video_url = st.text_input("영상 URL 입력", placeholder="https://...")

    col1, col2, col3 = st.columns(3)
    with col1:
        target_seconds = st.slider("하이라이트 길이(초)", 15, 180, 60)
        clip_min = st.slider("최소 클립 길이", 1.0, 8.0, 2.0, 0.5)
        clip_max = st.slider("최대 클립 길이", 2.0, 15.0, 8.0, 0.5)
    with col2:
        use_whisper = st.checkbox("Whisper 대사 분석 사용", value=False)
        whisper_model = st.selectbox("Whisper 모델", ["tiny", "base", "small", "medium"], index=2)
        make_shorts = st.checkbox("쇼츠도 같이 생성", value=True)
    with col3:
        shorts_count = st.slider("쇼츠 개수", 1, 5, 3)
        shorts_duration = st.slider("쇼츠 길이", 10, 60, 20)
        sample_width = st.slider("분석 해상도 폭", 160, 640, 320, 32)

    if uploaded is not None:
        st.video(uploaded)
    elif source_type == "영상 URL" and video_url.strip():
        st.caption("URL 영상은 서버에서 먼저 다운로드한 뒤 분석합니다.")

    if st.button("하이라이트 만들기", type="primary"):
        if source_type == "파일 업로드" and uploaded is None:
            st.error("먼저 영상을 업로드하세요.")
            return
        if source_type == "영상 URL" and not video_url.strip():
            st.error("영상 URL을 입력하세요.")
            return

        with tempfile.TemporaryDirectory() as td:
            if source_type == "파일 업로드":
                input_path = os.path.join(td, uploaded.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded.read())
            else:
                with st.spinner("URL에서 영상을 다운로드 중입니다..."):
                    input_path = download_video_from_url(video_url.strip(), td)

            output_dir = os.path.join(td, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "highlight.mp4")

            cfg = Config(
                input_path=input_path,
                output_path=output_path,
                target_seconds=target_seconds,
                clip_min=clip_min,
                clip_max=clip_max,
                sample_width=sample_width,
                use_whisper=use_whisper,
                whisper_model=whisper_model,
                make_shorts=make_shorts,
                shorts_count=shorts_count,
                shorts_duration=shorts_duration,
            )

            try:
                with st.spinner("분석 중입니다..."):
                    result = build_highlight(cfg)

                st.success("완료되었습니다.")

                st.subheader("하이라이트 영상")
                with open(result["output_video"], "rb") as f:
                    data = f.read()
                st.video(data)
                st.download_button("하이라이트 다운로드", data, file_name="highlight.mp4", mime="video/mp4")

                st.subheader("선택된 구간")
                for i, seg in enumerate(result["segments"], 1):
                    st.write(
                        f"{i}. {sec_to_hhmmss(seg.start)} ~ {sec_to_hhmmss(seg.end)} | score={seg.score:.3f}"
                    )
                    if seg.reason:
                        st.caption(seg.reason)

                if result["shorts_outputs"]:
                    st.subheader("쇼츠 결과")
                    for path in result["shorts_outputs"]:
                        with open(path, "rb") as f:
                            d = f.read()
                        st.video(d)
                        st.download_button(
                            label=f"{os.path.basename(path)} 다운로드",
                            data=d,
                            file_name=os.path.basename(path),
                            mime="video/mp4",
                            key=path,
                        )

                with open(result["report_path"], "rb") as f:
                    report_data = f.read()
                st.download_button(
                    "리포트 JSON 다운로드",
                    report_data,
                    file_name="highlight_report.json",
                    mime="application/json",
                )
            except Exception as e:
                st.exception(e)


# =========================
# CLI
# =========================
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Video Highlight AI - All in One")
    parser.add_argument("--input", help="입력 영상 경로")
    parser.add_argument("--output", help="출력 하이라이트 경로")
    parser.add_argument("--target-seconds", type=int, default=60)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--clip-min", type=float, default=2.0)
    parser.add_argument("--clip-max", type=float, default=8.0)
    parser.add_argument("--merge-gap", type=float, default=1.2)
    parser.add_argument("--sample-width", type=int, default=320)
    parser.add_argument("--audio-weight", type=float, default=0.30)
    parser.add_argument("--motion-weight", type=float, default=0.30)
    parser.add_argument("--scene-weight", type=float, default=0.20)
    parser.add_argument("--speech-weight", type=float, default=0.20)
    parser.add_argument("--use-whisper", action="store_true")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--make-shorts", action="store_true")
    parser.add_argument("--shorts-count", type=int, default=3)
    parser.add_argument("--shorts-duration", type=int, default=20)
    parser.add_argument("--streamlit", action="store_true", help="streamlit GUI 실행")
    return parser.parse_args()


# =========================
# Entry Point
# =========================
def main():
    args = parse_args()

    if args.streamlit:
        run_streamlit_app()
        return

    if not args.input or not args.output:
        print("입력/출력 경로가 없어서 GUI 실행 안내를 표시합니다.")
        print("GUI 실행:")
        print("  streamlit run video_highlight_ai_all_in_one.py")
        print("CLI 실행:")
        print("  python video_highlight_ai_all_in_one.py --input input.mp4 --output outputs/highlight.mp4 --use-whisper --make-shorts")
        return

    cfg = Config(
        input_path=args.input,
        output_path=args.output,
        target_seconds=args.target_seconds,
        window_seconds=args.window_seconds,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        merge_gap=args.merge_gap,
        sample_width=args.sample_width,
        audio_weight=args.audio_weight,
        motion_weight=args.motion_weight,
        scene_weight=args.scene_weight,
        speech_weight=args.speech_weight,
        use_whisper=args.use_whisper,
        whisper_model=args.whisper_model,
        make_shorts=args.make_shorts,
        shorts_count=args.shorts_count,
        shorts_duration=args.shorts_duration,
    )

    result = build_highlight(cfg)

    print("\n완료")
    print(f"하이라이트 영상: {result['output_video']}")
    print(f"리포트: {result['report_path']}")
    if result["shorts_outputs"]:
        print("쇼츠 결과:")
        for p in result["shorts_outputs"]:
            print(f"- {p}")


if __name__ == "__main__":
    run_streamlit_app()
